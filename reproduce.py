"""Train the paper models and reproduce the primary ASDSpeech results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
EXPERTS = {"rrb_v": ROOT / "configs" / "rrb_v.yaml",
           "rrb_e": ROOT / "configs" / "rrb_e.yaml"}
DEFAULT_SEEDS = (2024, 2025, 2026, 2027, 2028)
TIMEPOINTS = ("T1", "T2")
PREDICTION_COLUMNS = (
    "recording_id", "sa_true", "sa_pred", "sa_std", "rrb_true",
    "rrb_pred", "rrb_std", "total_true", "total_pred", "total_std",
)


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_pred, mean_true = np.mean(y_pred), np.mean(y_true)
    covariance = np.mean((y_pred - mean_pred) * (y_true - mean_true))
    denominator = np.var(y_pred) + np.var(y_true) + (mean_pred - mean_true) ** 2
    return float(2 * covariance / (denominator + 1e-8))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred).statistic)


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "pearson":
        return pearson(y_true, y_pred)
    if metric == "ccc":
        return ccc(y_true, y_pred)
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    raise ValueError(metric)


def validate_data(data_dir: Path) -> None:
    required = [data_dir / "train_data.mat"]
    required += [data_dir / f"data_{timepoint}.xlsx" for timepoint in TIMEPOINTS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ASDSpeech files: " + ", ".join(missing))

    data = loadmat(required[0], variable_names=["features", "num_matrices", "sa", "rrb"])
    num_matrices = int(np.squeeze(data["num_matrices"]))
    if num_matrices != 10 or data["features"].shape[0] != 136 * num_matrices:
        raise ValueError("Expected 136 development recordings with ten matrices each")
    if np.asarray(data["features"].reshape(-1)[0]).shape != (100, 49):
        raise ValueError("Expected 100 x 49 descriptor matrices")

    for timepoint in TIMEPOINTS:
        frame = pd.read_excel(data_dir / f"data_{timepoint}.xlsx", dtype={"rec_id": str})
        if len(frame) != 61 or frame["rec_id"].duplicated().any():
            raise ValueError(f"Expected 61 unique {timepoint} recordings")
        absent = [rec_id for rec_id in frame["rec_id"]
                  if not (data_dir / f"{rec_id}.mat").is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {timepoint} matrices for {absent[:5]}")


def configured_run(expert: str, seed: int, data_dir: Path, output_dir: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(EXPERTS[expert].read_text(encoding="utf-8"))
    run_base = output_dir / "models" / expert / f"seed_{seed}"
    config["paths_config"]["data_file_path"] = str(data_dir / "train_data.mat")
    config["paths_config"]["output_dir"] = str(run_base)
    config_path = output_dir / "configs" / f"{expert}_seed_{seed}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config


def matching_run(run_base: Path, config: dict, seed: int) -> Path | None:
    candidates = sorted(run_base.glob("experiment_*"), key=lambda path: path.stat().st_mtime,
                        reverse=True)
    for run_dir in candidates:
        summary = run_dir / "summary.txt"
        saved_config = run_dir / "config.yaml"
        if not summary.is_file() or not saved_config.is_file():
            continue
        if f"Random seed: {seed}\n" not in summary.read_text(encoding="utf-8"):
            continue
        folds = range(1, int(config["params_config"]["k_folds"]) + 1)
        if not all(
            (run_dir / f"fold_{fold}" / name).is_file()
            for fold in folds
            for name in ("model_SA.pth", "model_RRB.pth", "val_pred_RRB.txt")
        ):
            continue
        if yaml.safe_load(saved_config.read_text(encoding="utf-8")) == config:
            return run_dir
    return None


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed (exit {result.returncode}); see {log_path}")


def train_experts(seeds: tuple[int, ...], data_dir: Path, output_dir: Path) -> dict:
    runs: dict[str, dict[int, Path]] = {expert: {} for expert in EXPERTS}
    for expert in EXPERTS:
        for seed in seeds:
            config_path, config = configured_run(expert, seed, data_dir, output_dir)
            run_base = Path(config["paths_config"]["output_dir"])
            run_dir = matching_run(run_base, config, seed)
            if run_dir is None:
                run_logged([sys.executable, str(ROOT / "src" / "train.py"),
                            "--config", str(config_path), "--seed", str(seed)],
                           output_dir / "logs" / f"{expert}_seed_{seed}_train.log")
                run_dir = matching_run(run_base, config, seed)
                if run_dir is None:
                    raise RuntimeError(f"Training did not produce a complete run: {run_base}")
            else:
                print(f"Reusing {expert}, seed {seed}: {run_dir}")
            runs[expert][seed] = run_dir
    return runs


def load_oof(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    blocks = [np.atleast_2d(np.loadtxt(run_dir / f"fold_{fold}" / "val_pred_RRB.txt"))
              for fold in range(1, 6)]
    if any(block.shape[1] != 2 for block in blocks):
        raise ValueError(f"Malformed OOF predictions in {run_dir}")
    values = np.concatenate(blocks)
    return values[:, 0], values[:, 1]


def select_weight(runs: dict, seeds: tuple[int, ...], output_dir: Path) -> float:
    rows = []
    for seed in seeds:
        y_v, pred_v = load_oof(runs["rrb_v"][seed])
        y_e, pred_e = load_oof(runs["rrb_e"][seed])
        if not np.array_equal(y_v, y_e):
            raise ValueError(f"RRB experts have different OOF fold order for seed {seed}")
        for step in range(11):
            error_weight = step / 10
            pred = (1 - error_weight) * pred_v + error_weight * pred_e
            rows.append({"seed": seed, "rrb_e_weight": error_weight,
                         "pearson": pearson(y_v, pred), "ccc": ccc(y_v, pred),
                         "mae": score(y_v, pred, "mae")})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "oof_weight_scores.csv", index=False)
    mean_ccc = frame.groupby("rrb_e_weight", sort=True)["ccc"].mean()
    selected = float(mean_ccc.idxmax())
    (output_dir / "fusion_weight.json").write_text(json.dumps({
        "selection": "maximum mean development-set OOF CCC on 0.1 grid",
        "rrb_v_weight": 1 - selected,
        "rrb_e_weight": selected,
        "seeds": seeds,
        "benchmark_labels_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    print(f"Selected RRB-E weight from development OOF: {selected:.1f}")
    return selected


def evaluate_experts(runs: dict, seeds: tuple[int, ...], data_dir: Path,
                     output_dir: Path) -> None:
    for expert in EXPERTS:
        for seed in seeds:
            run_dir = runs[expert][seed]
            if all((run_dir / f"Pred_{timepoint}.txt").is_file() for timepoint in TIMEPOINTS):
                print(f"Reusing {expert}, seed {seed} evaluation")
                continue
            run_logged([sys.executable, str(ROOT / "src" / "evaluate.py"),
                        "--results_dir", str(run_dir), "--data-dir", str(data_dir)],
                       output_dir / "logs" / f"{expert}_seed_{seed}_evaluate.log")


def read_predictions(run_dir: Path, timepoint: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / f"Pred_{timepoint}.txt", sep=r"\s+", comment="#",
                        names=PREDICTION_COLUMNS, dtype={"recording_id": str})
    if len(frame) != 61 or frame["recording_id"].duplicated().any():
        raise ValueError(f"Expected 61 unique predictions in {run_dir} at {timepoint}")
    return frame.set_index("recording_id").sort_index()


def benchmark_predictions(runs: dict, seeds: tuple[int, ...], weight: float,
                          timepoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = []
    for seed in seeds:
        v = read_predictions(runs["rrb_v"][seed], timepoint)
        e = read_predictions(runs["rrb_e"][seed], timepoint)
        if not v.index.equals(e.index):
            raise ValueError(f"Benchmark recording IDs differ for seed {seed}")
        for target in ("sa", "rrb", "total"):
            if not np.allclose(v[f"{target}_true"], e[f"{target}_true"]):
                raise ValueError(f"Benchmark {target} labels differ for seed {seed}")
        by_seed.append(pd.DataFrame({
            "sa_true": v["sa_true"], "rrb_true": v["rrb_true"],
            "total_true": v["total_true"], "sa_pred": v["sa_pred"],
            "rrb_v_pred": v["rrb_pred"], "rrb_e_pred": e["rrb_pred"],
            "rrb_pred": (1 - weight) * v["rrb_pred"] + weight * e["rrb_pred"],
        }, index=v.index))
    reference = by_seed[0]
    for frame in by_seed[1:]:
        if not reference.index.equals(frame.index):
            raise ValueError(f"Recording IDs differ across seeds at {timepoint}")
        for target in ("sa", "rrb", "total"):
            if not np.allclose(reference[f"{target}_true"], frame[f"{target}_true"]):
                raise ValueError(f"Benchmark labels differ across seeds at {timepoint}")
    final = reference[["sa_true", "rrb_true", "total_true"]].copy()
    for column in ("sa_pred", "rrb_v_pred", "rrb_e_pred", "rrb_pred"):
        final[column] = np.mean(np.stack([frame[column].to_numpy() for frame in by_seed]), axis=0)
    final["total_pred"] = final["sa_pred"] + final["rrb_pred"]
    baseline = final[["sa_true", "rrb_true", "total_true", "sa_pred", "rrb_v_pred"]].copy()
    baseline["rrb_pred"] = baseline["rrb_v_pred"]
    baseline["total_pred"] = baseline["sa_pred"] + baseline["rrb_pred"]
    return final, baseline


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 1.0
    for rank, index in enumerate(order[::-1], start=1):
        previous = min(previous, p_values[index] * len(p_values) / (len(p_values) - rank + 1))
        adjusted[index] = previous
    return adjusted.tolist()


def paired_bootstrap(final_by_timepoint: dict, baseline_by_timepoint: dict,
                     n_boot: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(20260723)
    rows = []
    for timepoint in TIMEPOINTS:
        final = final_by_timepoint[timepoint]
        baseline = baseline_by_timepoint[timepoint]
        for target in ("sa", "rrb", "total"):
            truth = final[f"{target}_true"].to_numpy(float)
            candidate = final[f"{target}_pred"].to_numpy(float)
            original = baseline[f"{target}_pred"].to_numpy(float)
            for metric in ("pearson", "ccc", "mae"):
                observed = score(truth, candidate, metric) - score(truth, original, metric)
                deltas = np.empty(n_boot)
                for iteration in range(n_boot):
                    indices = rng.integers(0, len(truth), size=len(truth))
                    deltas[iteration] = (
                        score(truth[indices], candidate[indices], metric)
                        - score(truth[indices], original[indices], metric)
                    )
                low, high = np.percentile(deltas, [2.5, 97.5])
                p_value = min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
                rows.append({"timepoint": timepoint, "target": target.upper(),
                             "metric": metric, "fusion_minus_rrb_v": observed,
                             "ci95_low": low, "ci95_high": high, "bootstrap_p": p_value})
    result = pd.DataFrame(rows)
    result["rrb_six_effect_fdr_p"] = np.nan
    rrb_indices = result.index[result["target"] == "RRB"]
    result.loc[rrb_indices, "rrb_six_effect_fdr_p"] = bh_adjust(
        result.loc[rrb_indices, "bootstrap_p"].tolist())
    return result


def write_results(runs: dict, seeds: tuple[int, ...], weight: float,
                  output_dir: Path) -> None:
    final_by_timepoint = {}
    baseline_by_timepoint = {}
    metric_rows = []
    for timepoint in TIMEPOINTS:
        final, baseline = benchmark_predictions(runs, seeds, weight, timepoint)
        final_by_timepoint[timepoint] = final
        baseline_by_timepoint[timepoint] = baseline
        final.reset_index().to_csv(output_dir / f"predictions_{timepoint}.csv", index=False)
        for model, target, prediction in (
            ("SA", "sa", "sa_pred"), ("RRB-V", "rrb", "rrb_v_pred"),
            ("RRB-E", "rrb", "rrb_e_pred"), ("RRB-fusion", "rrb", "rrb_pred"),
            ("Total", "total", "total_pred"),
        ):
            truth = final[f"{target}_true"].to_numpy(float)
            pred = final[prediction].to_numpy(float)
            metric_rows.append({"timepoint": timepoint, "model": model, "n": len(truth),
                                **{name: score(truth, pred, name)
                                   for name in ("pearson", "ccc", "mae", "rmse")}})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    paired_bootstrap(final_by_timepoint, baseline_by_timepoint).to_csv(
        output_dir / "paired_rrb_fusion.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing the original ASDSpeech data files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct integers")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    validate_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = train_experts(seeds, data_dir, output_dir)
    weight = select_weight(runs, seeds, output_dir)
    evaluate_experts(runs, seeds, data_dir, output_dir)
    write_results(runs, seeds, weight, output_dir)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()

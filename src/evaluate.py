"""
Unified Evaluation Script for ASD Speech Prediction.

Evaluates trained SA (regression) and RRB (ordinal) models on T1/T2 datasets.

Usage:
    python evaluate.py -r ../../results/experiment_XXXXXXXX
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.io import loadmat
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import os
import shutil

from train import SA_Model, RRB_Model, MultiTaskMILModel, set_seed


class _SAFromMTL(torch.nn.Module):
    """Wrap MultiTaskMILModel to behave like SA_Model (forward -> (B,1))."""
    def __init__(self, mtl: MultiTaskMILModel):
        super().__init__()
        self.mtl = mtl

    def forward(self, X):
        sa, _ = self.mtl(X)
        return sa


class _RRBFromMTL(torch.nn.Module):
    """Wrap MultiTaskMILModel to behave like RRB_Model (predict_continuous)."""
    def __init__(self, mtl: MultiTaskMILModel):
        super().__init__()
        self.mtl = mtl

    def predict_continuous(self, X):
        _, logits = self.mtl(X)
        return self.mtl.rrb_predict_continuous(logits)


def load_sa_model(model_path, params, device):
    """Load a scalar-regression or cumulative-threshold SA model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ck_params = dict(params)
    if isinstance(checkpoint.get('model_kwargs'), dict):
        ck_params.update(checkpoint['model_kwargs'])
    
    model = SA_Model(
        emb_dim=ck_params.get('emb_dim', 128),
        dropout=ck_params.get('dropout', 0.2),
        encoder_variant=ck_params.get('encoder_variant', 'tcn_stats'),
        pool_variant=ck_params.get('pool_variant', 'gated'),
        sa_head_variant=ck_params.get('sa_head_variant', 'regression'),
        sa_num_classes=int(ck_params.get('sa_num_classes', 23)),
        sa_score_min=float(ck_params.get('sa_score_min', 0.0)),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.matrix_level_supervision = bool(ck_params.get('matrix_level_supervision', False))
    model.eval()
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.var_ = scaler.scale_ ** 2
    
    return model, scaler, checkpoint.get('metrics', {})


def load_rrb_model(model_path, params, device):
    """Load RRB ordinal model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ck_params = dict(params)
    if isinstance(checkpoint.get('model_kwargs'), dict):
        ck_params.update(checkpoint['model_kwargs'])
    
    num_classes = checkpoint.get('num_classes', 9)
    
    model = RRB_Model(
        emb_dim=ck_params.get('emb_dim', 128),
        dropout=ck_params.get('dropout', 0.2),
        num_classes=num_classes,
        encoder_variant=ck_params.get('encoder_variant', 'tcn_stats'),
        pool_variant=ck_params.get('pool_variant', 'gated'),
        rrb_pool_variant=ck_params.get('rrb_pool_variant', ''),
        feat_group_dropout=0.0,
        rrb_head_variant=ck_params.get('rrb_head_variant', 'independent'),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.matrix_level_supervision = bool(ck_params.get('matrix_level_supervision', False))
    model.eval()
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.var_ = scaler.scale_ ** 2
    
    return model, scaler, checkpoint.get('metrics', {})


def load_mtl_model(model_path, params, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ck_params = dict(params)
    if isinstance(checkpoint.get('model_kwargs'), dict):
        ck_params.update(checkpoint['model_kwargs'])
    num_classes = checkpoint.get('num_classes', 9)
    model = MultiTaskMILModel(
        emb_dim=ck_params.get('emb_dim', 128),
        dropout=ck_params.get('dropout', 0.2),
        num_classes=num_classes,
        encoder_variant=ck_params.get('encoder_variant', 'tcn_stats'),
        pool_variant=ck_params.get('pool_variant', 'gated'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.var_ = scaler.scale_ ** 2
    return model, scaler, checkpoint


def bundle_models_into_results(results_dir: Path, sa_results_dir: Path, rrb_results_dir: Path):
    results_dir = Path(results_dir).resolve()
    sa_results_dir = Path(sa_results_dir).resolve()
    rrb_results_dir = Path(rrb_results_dir).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        src_sa = sa_results_dir / f'fold_{fold}' / 'model_SA.pth'
        src_rrb = rrb_results_dir / f'fold_{fold}' / 'model_RRB.pth'
        dst_fold = results_dir / f'fold_{fold}'
        dst_fold.mkdir(parents=True, exist_ok=True)
        if src_sa.exists():
            shutil.copy2(src_sa, dst_fold / 'model_SA.pth')
        else:
            print(f"Warning: missing SA fold_{fold}: {src_sa}")
        if src_rrb.exists():
            shutil.copy2(src_rrb, dst_fold / 'model_RRB.pth')
        else:
            print(f"Warning: missing RRB fold_{fold}: {src_rrb}")

    for src_cfg in (sa_results_dir / 'config.yaml', rrb_results_dir / 'config.yaml'):
        if src_cfg.exists():
            shutil.copy2(src_cfg, results_dir / 'config.yaml')
            break


def calculate_metrics(y_true, y_pred, is_ordinal=False, y_range_override=None):
    """Calculate evaluation metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    if y_range_override is None:
        y_range = np.max(y_true) - np.min(y_true)
    else:
        y_range = float(y_range_override)
    nrmse = rmse / y_range if y_range > 0 else 0
    
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        r, p = 0.0, 1.0
        r_spear, p_spear = 0.0, 1.0
    else:
        r, p = stats.pearsonr(y_pred, y_true)
        r_spear, p_spear = stats.spearmanr(y_pred, y_true)
    
    mx, my = np.mean(y_pred), np.mean(y_true)
    sx2, sy2 = np.var(y_pred), np.var(y_true)
    sxy = np.mean((y_pred - mx) * (y_true - my))
    ccc = (2 * sxy) / (sx2 + sy2 + (mx - my)**2 + 1e-8)
    
    metrics = {
        'RMSE': float(rmse), 'MAE': float(mae), 'NRMSE': float(nrmse),
        'R': float(r) if not np.isnan(r) else 0.0,
        'R_spear': float(r_spear) if not np.isnan(r_spear) else 0.0,
        'CCC': float(ccc) if not np.isnan(ccc) else 0.0,
        'p': float(p) if not np.isnan(p) else 1.0,
    }
    
    if is_ordinal:
        metrics['Accuracy'] = float(np.mean(np.round(y_pred) == y_true))
    
    return metrics


def predict_recording(mat_path, sa_models, sa_scalers, rrb_models, rrb_scalers, device, mtl_models=None, mtl_scalers=None):
    """Predict SA/RRB (and Total=SA+RRB) for a single recording using ensemble."""
    data = loadmat(mat_path)
    
    # Load features
    raw_features = data['features']
    if raw_features.dtype == 'O':
        feats_list = []
        for i in range(raw_features.shape[0]):
            feat = raw_features[i, 0] if raw_features.ndim == 2 else raw_features[i]
            feats_list.append(np.array(feat))
        features = np.stack(feats_list)
    else:
        features = raw_features
    
    features = np.nan_to_num(features)
    if features.ndim == 2:
        features = features[np.newaxis, ...]
    
    # Prefer MTL models if provided; fallback to separate SA/RRB models
    all_preds_sa, all_preds_rrb, all_preds_total = [], [], []

    if mtl_models is not None and mtl_scalers is not None and len(mtl_models) > 0:
        for mtl, sc in zip(mtl_models, mtl_scalers):
            X = sc.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, M, 100, 49)
            with torch.no_grad():
                sa, rrb_logits = mtl(X_t)
                rrb = mtl.rrb_predict_continuous(rrb_logits).cpu().numpy().flatten()[0]
                sa = sa.cpu().numpy().flatten()[0]
            total = sa + rrb
            all_preds_sa.append(sa)
            all_preds_rrb.append(rrb)
            all_preds_total.append(total)
    else:
        if len(rrb_models) == 0 or len(rrb_scalers) == 0:
            for sa_model, sa_scaler in zip(sa_models, sa_scalers):
                X_sa = sa_scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
                X_sa_t = torch.tensor(X_sa, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    sa = sa_model(X_sa_t).cpu().numpy().flatten()[0]
                all_preds_sa.append(sa)
        else:
            for sa_model, sa_scaler, rrb_model, rrb_scaler in zip(sa_models, sa_scalers, rrb_models, rrb_scalers):
                X_sa = sa_scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
                X_rrb = rrb_scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
                X_sa_t = torch.tensor(X_sa, dtype=torch.float32).unsqueeze(0).to(device)
                X_rrb_t = torch.tensor(X_rrb, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    if bool(getattr(sa_model, "matrix_level_supervision", False)):
                        sa = sa_model(X_sa_t.squeeze(0)).mean().cpu().numpy().item()
                    else:
                        sa = sa_model(X_sa_t).cpu().numpy().flatten()[0]
                    if bool(getattr(rrb_model, "matrix_level_supervision", False)):
                        rrb = rrb_model.predict_continuous(X_rrb_t.squeeze(0)).mean().cpu().numpy().item()
                    else:
                        rrb = rrb_model.predict_continuous(X_rrb_t).cpu().numpy().flatten()[0]
                total = sa + rrb
                all_preds_sa.append(sa)
                all_preds_rrb.append(rrb)
                all_preds_total.append(total)

    if len(all_preds_rrb) == 0:
        return (
            float(np.mean(all_preds_sa)), float(np.std(all_preds_sa)),
            None, None,
            None, None,
        )
    return (
        float(np.mean(all_preds_sa)), float(np.std(all_preds_sa)),
        float(np.mean(all_preds_rrb)), float(np.std(all_preds_rrb)),
        float(np.mean(all_preds_total)), float(np.std(all_preds_total)),
    )


def evaluate_dataset(dataset_name, data_dir, results_dir, 
                     sa_models, sa_scalers, rrb_models, rrb_scalers, device,
                     params,
                     mtl_models=None, mtl_scalers=None):
    """Evaluate on T1 or T2 dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")
    
    meta_file = data_dir / f'data_{dataset_name}.xlsx'
    if not meta_file.exists():
        print(f"Warning: {meta_file} not found")
        return None
    
    df = pd.read_excel(meta_file)
    
    predictions = []
    true_sa, true_rrb = [], []
    true_total = []
    
    for idx, row in df.iterrows():
        rec_id = row['rec_id']
        mat_file = data_dir / f"{rec_id}.mat"
        
        if not mat_file.exists():
            print(f"  Warning: {mat_file} not found")
            continue
        
        pred_sa, std_sa, pred_rrb, std_rrb, pred_total, std_total = predict_recording(
            mat_file, sa_models, sa_scalers, rrb_models, rrb_scalers, device,
            mtl_models=mtl_models, mtl_scalers=mtl_scalers
        )
        has_rrb = pred_rrb is not None

        item = {
            'rec_id': rec_id,
            'true_sa': row['SA'], 'pred_sa': pred_sa, 'std_sa': std_sa,
        }
        if has_rrb:
            item.update({
                'true_rrb': row['RRB'], 'pred_rrb': pred_rrb, 'std_rrb': std_rrb,
                'true_total': float(row['SA']) + float(row['RRB']),
                'pred_total': pred_total, 'std_total': std_total
            })
        predictions.append(item)
        
        true_sa.append(row['SA'])
        if has_rrb:
            true_rrb.append(row['RRB'])
            true_total.append(float(row['SA']) + float(row['RRB']))
    
    pred_sa = [p['pred_sa'] for p in predictions]
    has_rrb = len(predictions) > 0 and ('pred_rrb' in predictions[0])
    if has_rrb:
        pred_rrb = [p['pred_rrb'] for p in predictions]
        pred_total = [p['pred_total'] for p in predictions]
    
    sa_range = float(params.get('sa_score_max', 22.0)) - float(params.get('sa_score_min', 0.0))
    rrb_range = float(params.get('rrb_score_max', 8.0)) - float(params.get('rrb_score_min', 0.0))
    total_range = sa_range + rrb_range

    metrics_sa = calculate_metrics(true_sa, pred_sa, y_range_override=sa_range)
    if has_rrb:
        metrics_rrb = calculate_metrics(true_rrb, pred_rrb, is_ordinal=True, y_range_override=rrb_range)
        metrics_total = calculate_metrics(true_total, pred_total, y_range_override=total_range)
    
    print(f"\n{dataset_name} Results:")
    print(f"  SA:  R={metrics_sa['R']:.4f}, CCC={metrics_sa['CCC']:.4f}, "
          f"RMSE={metrics_sa['RMSE']:.2f}, p={metrics_sa['p']:.4f}")
    if has_rrb:
        print(f"  RRB: R={metrics_rrb['R']:.4f}, CCC={metrics_rrb['CCC']:.4f}, "
              f"RMSE={metrics_rrb['RMSE']:.2f}, p={metrics_rrb['p']:.4f}, "
              f"Acc={metrics_rrb['Accuracy']:.2%}")
        print(f"  Total: R={metrics_total['R']:.4f}, CCC={metrics_total['CCC']:.4f}, "
              f"RMSE={metrics_total['RMSE']:.2f}, p={metrics_total['p']:.4f}")
    
    # Save predictions
    with open(results_dir / f'Pred_{dataset_name}.txt', 'w') as f:
        f.write(f"# {dataset_name} Predictions\n")
        if has_rrb:
            f.write("# rec_id true_SA pred_SA std_SA true_RRB pred_RRB std_RRB true_Total pred_Total std_Total\n")
            for p in predictions:
                f.write(f"{p['rec_id']} {p['true_sa']:.2f} {p['pred_sa']:.4f} {p['std_sa']:.4f} "
                       f"{p['true_rrb']:.0f} {p['pred_rrb']:.4f} {p['std_rrb']:.4f} "
                       f"{p['true_total']:.2f} {p['pred_total']:.4f} {p['std_total']:.4f}\n")
        else:
            f.write("# rec_id true_SA pred_SA std_SA\n")
            for p in predictions:
                f.write(f"{p['rec_id']} {p['true_sa']:.2f} {p['pred_sa']:.4f} {p['std_sa']:.4f}\n")
    
    # Save metrics
    with open(results_dir / f'Metrics_{dataset_name}.txt', 'w') as f:
        f.write(f"Metrics for {dataset_name}\n")
        f.write("="*50 + "\n\n")
        f.write("SA Prediction:\n")
        for k, v in metrics_sa.items():
            f.write(f"  {k}: {v:.4f}\n")
        if has_rrb:
            f.write("\nRRB Prediction (Ordinal):\n")
            for k, v in metrics_rrb.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\nTotal Prediction (SA+RRB):\n")
            for k, v in metrics_total.items():
                f.write(f"  {k}: {v:.4f}\n")
    
    # Plot
    if has_rrb:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    
    # SA plot
    ax = axes[0] if has_rrb else axes
    ax.scatter(true_sa, pred_sa, alpha=0.6, s=60, c='#1f77b4', edgecolors='white')
    ax.plot([min(true_sa), max(true_sa)], [min(true_sa), max(true_sa)], 'r--', lw=2)
    if np.std(pred_sa) > 0:
        slope, intercept, r_val, _, _ = stats.linregress(true_sa, pred_sa)
        x_line = np.linspace(min(true_sa), max(true_sa), 100)
        ax.plot(x_line, slope * x_line + intercept, 'g-', lw=2, alpha=0.7)
    ax.set_xlabel('True SA', fontsize=12)
    ax.set_ylabel('Predicted SA', fontsize=12)
    ax.set_title(f'{dataset_name} - SA\nR={metrics_sa["R"]:.3f}, CCC={metrics_sa["CCC"]:.3f}, p={metrics_sa["p"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # RRB plot
    if has_rrb:
        ax = axes[1]
        ax.scatter(true_rrb, pred_rrb, alpha=0.6, s=60, c='#ff7f0e', edgecolors='white')
        ax.plot([min(true_rrb), max(true_rrb)], [min(true_rrb), max(true_rrb)], 'r--', lw=2)
        if np.std(pred_rrb) > 0:
            slope, intercept, r_val, _, _ = stats.linregress(true_rrb, pred_rrb)
            x_line = np.linspace(min(true_rrb), max(true_rrb), 100)
            ax.plot(x_line, slope * x_line + intercept, 'g-', lw=2, alpha=0.7)
        ax.set_xlabel('True RRB', fontsize=12)
        ax.set_ylabel('Predicted RRB', fontsize=12)
        ax.set_title(f'{dataset_name} - RRB (Ordinal)\nR={metrics_rrb["R"]:.3f}, CCC={metrics_rrb["CCC"]:.3f}, p={metrics_rrb["p"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{dataset_name}_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: Pred_{dataset_name}.txt, Metrics_{dataset_name}.txt, {dataset_name}_scatter.png")
    
    if has_rrb:
        return {'SA': metrics_sa, 'RRB': metrics_rrb}
    return {'SA': metrics_sa}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir', required=True,
                        help='Path to results directory from train.py')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Directory containing the released ASDSpeech data files')
    parser.add_argument('--sa_results_dir', default=None,
                        help='Optional: load SA models from a different results directory')
    parser.add_argument('--rrb_results_dir', default=None,
                        help='Optional: load RRB models from a different results directory')
    parser.add_argument('--bundle_copy', action='store_true',
                        help='Copy SA/RRB fold checkpoints into results_dir before evaluating')
    args = parser.parse_args()
    
    set_seed(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {results_dir}")

    sa_results_dir = Path(args.sa_results_dir).resolve() if args.sa_results_dir else results_dir
    rrb_results_dir = Path(args.rrb_results_dir).resolve() if args.rrb_results_dir else results_dir
    if args.bundle_copy and ((sa_results_dir != results_dir) or (rrb_results_dir != results_dir)):
        print(f"Bundling SA from:  {sa_results_dir}")
        print(f"Bundling RRB from: {rrb_results_dir}")
        bundle_models_into_results(results_dir, sa_results_dir, rrb_results_dir)
        sa_results_dir = results_dir
        rrb_results_dir = results_dir
    if args.sa_results_dir or args.rrb_results_dir:
        print(f"SA models from:  {sa_results_dir}")
        print(f"RRB models from: {rrb_results_dir}")
    
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    data_dir = args.data_dir.resolve() if args.data_dir else root_dir / 'data'
    print(f"Data directory: {data_dir}")
    
    params = {'emb_dim': 128, 'dropout': 0.2, 'encoder_variant': 'tcn_stats', 'pool_variant': 'gated'}
    cfg_path = results_dir / 'config.yaml'
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and isinstance(cfg.get('params_config'), dict):
                params.update(cfg['params_config'])
        except Exception:
            pass
    
    sa_models, sa_scalers = [], []
    rrb_models, rrb_scalers = [], []
    mtl_models, mtl_scalers = [], []
    
    # Load models per fold; support mixing SA/RRB from different directories.
    # Priority order per fold:
    #   - If not mixing dirs: prefer MTL (model_MTL.pth)
    #   - If mixing dirs: load SA from sa_results_dir (model_SA.pth preferred), RRB from rrb_results_dir (model_RRB.pth preferred)
    for fold in range(1, 6):
        fold_dir_out = results_dir / f'fold_{fold}'
        fold_dir_sa = sa_results_dir / f'fold_{fold}'
        fold_dir_rrb = rrb_results_dir / f'fold_{fold}'

        # Ensure output fold dir exists for saving preds/plots
        fold_dir_out.mkdir(parents=True, exist_ok=True)

        mixing = (sa_results_dir != results_dir) or (rrb_results_dir != results_dir)

        if not mixing:
            # Prefer new multitask checkpoint if present
            mtl_path = (results_dir / f'fold_{fold}' / 'model_MTL.pth')
            if mtl_path.exists():
                mtl_model, mtl_scaler, ckpt = load_mtl_model(mtl_path, params, device)
                mtl_models.append(mtl_model)
                mtl_scalers.append(mtl_scaler)
                sa_r = ckpt.get('metrics_sa', {}).get('R', 0.0)
                rrb_r = ckpt.get('metrics_rrb', {}).get('R', 0.0)
                print(f"Loaded fold_{fold}: (MTL) SA R={sa_r:.4f}, RRB R={rrb_r:.4f}")
                continue

        # SA
        sa_path = fold_dir_sa / 'model_SA.pth'
        sa_mtl_path = fold_dir_sa / 'model_MTL.pth'
        if sa_path.exists():
            sa_model, sa_scaler, sa_metrics = load_sa_model(sa_path, params, device)
            sa_models.append(sa_model)
            sa_scalers.append(sa_scaler)
            print(f"Loaded fold_{fold}: SA (separate) R={sa_metrics.get('R', 0):.4f}")
        elif sa_mtl_path.exists():
            # Fallback: use SA head from MTL checkpoint
            mtl_model, mtl_scaler, ckpt = load_mtl_model(sa_mtl_path, params, device)
            sa_models.append(_SAFromMTL(mtl_model))
            sa_scalers.append(mtl_scaler)
            print(f"Loaded fold_{fold}: SA (from MTL) R={ckpt.get('metrics_sa', {}).get('R', 0.0):.4f}")
        else:
            print(f"Warning: SA model not found for fold_{fold} in {fold_dir_sa}")
            continue

        # RRB
        rrb_path = fold_dir_rrb / 'model_RRB.pth'
        rrb_mtl_path = fold_dir_rrb / 'model_MTL.pth'
        if rrb_path.exists():
            rrb_model, rrb_scaler, rrb_metrics = load_rrb_model(rrb_path, params, device)
            rrb_models.append(rrb_model)
            rrb_scalers.append(rrb_scaler)
            print(f"Loaded fold_{fold}: RRB (separate) R={rrb_metrics.get('R', 0):.4f}")
        elif rrb_mtl_path.exists():
            mtl_model, mtl_scaler, ckpt = load_mtl_model(rrb_mtl_path, params, device)
            rrb_models.append(_RRBFromMTL(mtl_model))
            rrb_scalers.append(mtl_scaler)
            print(f"Loaded fold_{fold}: RRB (from MTL) R={ckpt.get('metrics_rrb', {}).get('R', 0.0):.4f}")
        else:
            print(f"Warning: RRB model not found for fold_{fold} in {fold_dir_rrb}")
            continue
    
    if len(mtl_models) > 0:
        print(f"\nLoaded {len(mtl_models)} MTL models")
    else:
        # Align lengths to avoid zip truncation surprises
        n = min(len(sa_models), len(rrb_models))
        if n == 0:
            if len(sa_models) > 0:
                print(f"\nLoaded {len(sa_models)} SA models (SA-only evaluation)")
            elif len(rrb_models) > 0:
                print(f"\nLoaded {len(rrb_models)} RRB models (RRB-only evaluation)")
            else:
                raise RuntimeError("No models loaded. Please check results directories and fold contents.")
        else:
            sa_models, sa_scalers = sa_models[:n], sa_scalers[:n]
            rrb_models, rrb_scalers = rrb_models[:n], rrb_scalers[:n]
            print(f"\nLoaded {len(sa_models)} SA models and {len(rrb_models)} RRB models (paired)")
    
    # Evaluate on T1 and T2
    all_results = {}
    for dataset in ['T1', 'T2']:
        result = evaluate_dataset(
            dataset, data_dir, results_dir,
            sa_models, sa_scalers, rrb_models, rrb_scalers, device,
            params=params,
            mtl_models=mtl_models, mtl_scalers=mtl_scalers
        )
        if result:
            all_results[dataset] = result
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    
    for dataset, result in all_results.items():
        print(f"\n{dataset}:")
        print(f"  SA:  R={result['SA']['R']:.4f}, CCC={result['SA']['CCC']:.4f}, p={result['SA']['p']:.4f}")
        if 'RRB' in result:
            print(f"  RRB: R={result['RRB']['R']:.4f}, CCC={result['RRB']['CCC']:.4f}, p={result['RRB']['p']:.4f}")
    
    # Save final summary
    with open(results_dir / 'evaluation_summary.txt', 'w') as f:
        f.write("EVALUATION SUMMARY (T1/T2)\n")
        f.write("="*60 + "\n\n")
        for dataset, result in all_results.items():
            f.write(f"{dataset}:\n")
            f.write(f"  SA:  R={result['SA']['R']:.4f}, CCC={result['SA']['CCC']:.4f}, p={result['SA']['p']:.4f}\n")
            if 'RRB' in result:
                f.write(f"  RRB: R={result['RRB']['R']:.4f}, CCC={result['RRB']['CCC']:.4f}, p={result['RRB']['p']:.4f}\n\n")
            else:
                f.write("\n")


if __name__ == '__main__':
    main()

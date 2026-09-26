# Recording-level MIL for ADOS-2 score estimation

Code for estimating ADOS-2 Social Affect (SA), Restricted and Repetitive Behavior (RRB), and total scores from the released ASDSpeech speech descriptors. Each assessment is represented as a set of ten 100 x 49 descriptor matrices. The release trains one SA model and two ordinal RRB models (RRB-V and RRB-E); the RRB fusion weight is selected from development-set out-of-fold (OOF) predictions.

## Data

Download the public feature dataset from the [original ASDSpeech repository](https://github.com/Dinstein-Lab/ASDSpeech). This repository does not redistribute the data, the source study's code, raw audio, or trained checkpoints. Point `--data-dir` to the original repository's `data` folder. The required files are:

```text
data/
  train_data.mat
  data_T1.xlsx
  data_T2.xlsx
  <recording_id>.mat  (for every T1 and T2 recording)
```

The code uses the ten matrices present in the released `train_data.mat`, not the five-matrix subset described in the source article. T1 and T2 contain repeated assessments of the same 61 children.

## Environment

Tested with Python 3.13.2 and PyTorch 2.8.0 (CUDA 12.8) on Windows. A CPU installation also works, but training will be slower. From this directory:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA environment using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The remaining packages are listed in `requirements.txt`.

## Reproduce the primary results

Activate the virtual environment, then run:

```bash
python reproduce.py --data-dir /path/to/ASDSpeech/data
```

The default protocol trains both RRB variants for seeds 2024-2028 with five recording-level folds per seed. It selects one RRB-E weight on the 0.0-1.0 decile grid by maximizing mean development OOF concordance correlation coefficient, then evaluates the fixed mixture on T1 and T2. SA estimates come from the RRB-V run; total estimates are SA plus fused RRB. Completed runs are reused when the command is restarted.

The generated `outputs/` folder contains the model checkpoints and logs, `fusion_weight.json`, `oof_weight_scores.csv`, child-level `predictions_T1.csv` and `predictions_T2.csv`, `metrics.csv`, and `paired_rrb_fusion.csv` (10,000 paired bootstrap replicates, with false-discovery-rate adjustment across the six RRB effects). For a quick functional check, `--seeds 2024` runs one repetition; use all five default seeds to reproduce the paper protocol. Numerical values can vary slightly across hardware and library builds.

## Files

```text
reproduce.py       Training, OOF selection, evaluation, and result summaries
configs/           Final RRB-V and RRB-E training configurations
src/train.py       Recording-level models and five-fold training
src/evaluate.py    T1/T2 checkpoint inference
src/feature_groups.py  The nine descriptor-family index ranges
```

This is a research implementation for analysis of the released secondary descriptors, not a clinical assessment tool. The code is MIT-licensed; the ASDSpeech dataset remains subject to the original repository's terms.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a coursework repository for **INFO 6105** (Spring 2026) at Northeastern University. It contains problem sets and a final project as Jupyter notebooks, Python scripts, and PDF/Word documents.

## Repository Structure

- `problem_set1/` — Problem Set 1: Jupyter notebook (`PS-1-code.ipynb`) + PDF question/answer sheets
- `problem_set2/` — Problem Set 2 (empty)
- `midterm/` — Midterm project: house price prediction using Ridge Regression (`midterm_code&presentation.ipynb` + `.docx` report)
- `final/` — Final project: surface crack detection CNN (see below)

## Final Project (`final/`)

Binary image classification (Positive/Negative crack) on 227×227 RGB images using PyTorch. Runs on Northeastern's HPC cluster (SLURM + V100 GPU).

### Key files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters via argparse — single source of truth |
| `dataset.py` | `CrackDataset` (train, from `Positive/`/`Negative/` subdirs), `TestDataset` (flat dir), transforms |
| `model.py` | `CustomCNN` (configurable depth/filters) + pretrained ResNet18/34/50, VGG16 |
| `train.py` | Training loop, best-model checkpointing, history JSON export |
| `predict.py` | Load checkpoint → run inference → output `predictions.csv` |
| `analysis.py` | Bootstrap bias-variance decomp + training curve plots (single run) |
| `run_all_analysis.py` | BV analysis + plots for every checkpoint at once; saves comparison figures |

### SBATCH scripts (`final/scripts/`)

| Script | Purpose |
|--------|---------|
| `create_env.sbatch` | Create conda env (no GPU needed) |
| `check_env.sbatch` | Verify CUDA, PyTorch, and all packages |
| `train.sbatch` | Train one model; all hyperparams overridable via `--export` |
| `predict.sbatch` | Run inference; requires `CHECKPOINT=` env var |
| `bv_analysis.sbatch` | BV analysis for a single model config |
| `all_analysis.sbatch` | BV + plots for all `*_best.pt` checkpoints at once |
| `sweep.sh` | Submit a grid of training jobs (lr, dropout, pretrained models) |

### Cluster paths (Northeastern Discovery HPC)

- **Project**: `/home/shi.bow/p-self/course_project.INFO_6105/final`
- **Conda env**: `/home/shi.bow/p-self/conda_env/crack_detection`
- **SLURM working dir**: `/home/shi.bow/sbatchDir`
- **Logs**: `/home/shi.bow/sbatchDir/sbatchLog/crack_detection/`
- **Conda module**: `anaconda3/2024.06`
- **GPU**: `--gres=gpu:v100-sxm2:1`

### Data layout (not tracked in git)

```
final/data/
├── training/
│   ├── Positive/   *.jpg
│   └── Negative/   *.jpg
└── test/           *.png
```

### Key patterns

- Checkpoint files are saved as `checkpoints/<run_name>_best.pt` and always include `args` (the full hyperparameter dict) so any script can reconstruct the model without re-specifying flags.
- History JSON (`<run_name>_history.json`) stores `train_loss`, `train_acc`, `val_loss`, `val_acc` per epoch.
- `run_all_analysis.py` reads saved `args` from each checkpoint — never pass hyperparameters manually to it.
- Train images are JPG; test images are PNG. Both are handled identically via `PIL.Image.open().convert("RGB")`.
- Target label map: `Negative=0`, `Positive=1`. Prediction CSV columns: `image_id`, `predicted_class`.

## Midterm Project (`midterm/`)

Standard ML pipeline in a Jupyter notebook: data loading from Google Drive, EDA with seaborn pairplots, preprocessing (log-transform `price`, outlier removal, date → numeric, `yr_renovated` fill), Ridge regression with `RidgeCV` (5-fold CV), bias-variance analysis via `mlxtend.evaluate.bias_variance_decomp`, prediction export to CSV. `StandardScaler` is fitted on train and applied to test.

## Running Notebooks

```bash
jupyter notebook
```

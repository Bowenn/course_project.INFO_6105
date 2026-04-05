# Surface Crack Detection — Final Project (INFO 6105)

Binary image classification: detect whether a concrete surface image contains a crack (Positive) or not (Negative).

- **Training data**: 32,137 images, 227x227 RGB, ~16k per class
- **Metric**: Accuracy
- **Submission**: CSV with `image_id` and `predicted_class`

## Project Structure

```
final/
├── config.py          # All hyperparameters (argparse)
├── dataset.py         # Dataset classes and transforms
├── model.py           # CustomCNN + pretrained ResNet/VGG
├── train.py           # Training loop with checkpointing
├── predict.py         # Inference → CSV
├── analysis.py        # Bias-variance decomposition & training curve plots
├── environment.yml    # Conda environment spec
├── scripts/
│   ├── train.sbatch       # Training job
│   ├── predict.sbatch     # Prediction job
│   ├── bv_analysis.sbatch # Bias-variance analysis job
│   └── sweep.sh           # Submit hyperparameter sweep
└── data/                  # (not tracked in git)
    ├── train/
    │   ├── Positive/
    │   └── Negative/
    └── test/
        └── *.jpg
```

## Setup on HPC Cluster

### 1. Create conda environment

```bash
module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda env create -f environment.yml -p /home/shi.bow/p-self/conda_env/crack_detection
```

### 2. Prepare data

Place the dataset under `final/data/` so that training images are in `data/train/Positive/` and `data/train/Negative/`, and test images are in `data/test/`.

### 3. Create log directory

```bash
mkdir -p /home/shi.bow/sbatchDir/sbatchLog
```

## Training

### Single run (default: custom CNN, lr=0.001, batch_size=64, 20 epochs)

```bash
sbatch scripts/train.sbatch
```

### Custom hyperparameters

```bash
sbatch --export=ALL,MODEL=custom_cnn,LR=0.0001,EPOCHS=30,DROPOUT=0.3,RUN_NAME=cnn_lr0.0001_drop0.3 \
    scripts/train.sbatch
```

### Pretrained models

```bash
sbatch --export=ALL,MODEL=resnet18,LR=0.0001,PRETRAINED_FLAG="--pretrained",RUN_NAME=resnet18_pt \
    scripts/train.sbatch
```

### Hyperparameter sweep

```bash
bash scripts/sweep.sh
```

This submits jobs for:
- Learning rates: 0.01, 0.001, 0.0001
- Dropout: 0.3, 0.5, 0.7
- Pretrained ResNet18/34/50

## Prediction

```bash
sbatch --export=ALL,CHECKPOINT=checkpoints/custom_cnn_lr0.001_bs64_best.pt \
    scripts/predict.sbatch
```

Output: `predictions.csv` with columns `image_id`, `predicted_class`.

## Bias-Variance Analysis

```bash
sbatch --export=ALL,BV_RUNS=10,EPOCHS=10 scripts/bv_analysis.sbatch
```

Trains the model N times on random subsets and reports bias^2, variance, and average error on a held-out set.

## Plot Training Curves

After training, plot loss/accuracy curves from the saved history:

```bash
python analysis.py --checkpoint checkpoints/custom_cnn_lr0.001_bs64_best.pt
```

Outputs `training_curves.png`.

## Available Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `custom_cnn` | `custom_cnn`, `resnet18`, `resnet34`, `resnet50`, `vgg16` |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `64` | Batch size |
| `--epochs` | `20` | Training epochs |
| `--dropout` | `0.5` | Dropout rate |
| `--optimizer` | `adam` | `adam`, `sgd`, `adamw` |
| `--scheduler` | `cosine` | `cosine`, `step`, `none` |
| `--num_conv_blocks` | `4` | Conv blocks in custom CNN |
| `--base_filters` | `32` | Filters in first conv layer (doubles each block) |
| `--weight_decay` | `0.0001` | L2 regularization |
| `--pretrained` | off | Use ImageNet pretrained weights (resnet/vgg only) |
| `--val_ratio` | `0.2` | Validation split ratio |
| `--seed` | `42` | Random seed |
| `--bv_runs` | `0` | Bootstrap runs for bias-variance (0 = skip) |
| `--bv_train_ratio` | `0.5` | Data fraction per bootstrap sample |

## Monitoring Jobs

```bash
squeue -u $USER          # check running jobs
scancel <job_id>         # cancel a job
cat /home/shi.bow/sbatchDir/sbatchLog/train_<job_id>.txt   # view output
```

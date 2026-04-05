#!/bin/bash
# Submit multiple training jobs with different hyperparameters.
# Usage: bash scripts/sweep.sh

# Custom CNN with different learning rates
for LR in 0.01 0.001 0.0001; do
    sbatch --export=ALL,MODEL=custom_cnn,LR=${LR},RUN_NAME=cnn_lr${LR} \
        scripts/train.sbatch
done

# Custom CNN with different dropout rates
for DROP in 0.3 0.5 0.7; do
    sbatch --export=ALL,MODEL=custom_cnn,DROPOUT=${DROP},RUN_NAME=cnn_drop${DROP} \
        scripts/train.sbatch
done

# Pretrained models
for MODEL in resnet18 resnet34 resnet50; do
    sbatch --export=ALL,MODEL=${MODEL},LR=0.0001,PRETRAINED_FLAG="--pretrained",RUN_NAME=${MODEL}_pretrained \
        scripts/train.sbatch
done

echo "All sweep jobs submitted. Check with: squeue -u \$USER"

#!/bin/bash
#SBATCH -J crack_bv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/bv_%j.txt
#SBATCH -e sbatchLog/crack_detection/bv_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /home/shi.bow/p-self/conda_env/crack_detection

START_TIME=$(date +%s)
echo "job start: $(date)"

cd /home/shi.bow/p-self/course_project.INFO_6105/final

# ---- Hyperparameters ----
MODEL=${MODEL:-custom_cnn}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-10}
BV_RUNS=${BV_RUNS:-10}
BV_TRAIN_RATIO=${BV_TRAIN_RATIO:-0.5}
DROPOUT=${DROPOUT:-0.5}

echo "=== Bias-Variance Analysis ==="
echo "MODEL=${MODEL} LR=${LR} BV_RUNS=${BV_RUNS} EPOCHS=${EPOCHS}"
echo "==============================="

python analysis.py \
    --model ${MODEL} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dropout ${DROPOUT} \
    --bv_runs ${BV_RUNS} \
    --bv_train_ratio ${BV_TRAIN_RATIO} \
    --num_workers ${SLURM_CPUS_PER_TASK:-4}

conda deactivate
END_TIME=$(date +%s)
echo "job finished: $(date)"
echo "duration: $(( (END_TIME - START_TIME) / 3600 ))h $(( (END_TIME - START_TIME) % 3600 / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"

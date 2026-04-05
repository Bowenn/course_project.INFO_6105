#!/bin/bash
#SBATCH -J crack_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/train_%j.txt
#SBATCH -e sbatchLog/crack_detection/train_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /home/shi.bow/p-self/conda_env/crack_detection

echo "job start"

cd /home/shi.bow/p-self/course_project.INFO_6105/final
mkdir -p checkpoints

# ---- Hyperparameters (override via: sbatch --export=ALL,MODEL=resnet18,LR=0.0001 scripts/train.sbatch) ----
MODEL=${MODEL:-custom_cnn}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-20}
DROPOUT=${DROPOUT:-0.5}
OPTIMIZER=${OPTIMIZER:-adam}
SCHEDULER=${SCHEDULER:-cosine}
NUM_CONV_BLOCKS=${NUM_CONV_BLOCKS:-4}
BASE_FILTERS=${BASE_FILTERS:-32}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
RUN_NAME=${RUN_NAME:-"${MODEL}_lr${LR}_bs${BATCH_SIZE}"}
PRETRAINED_FLAG=${PRETRAINED_FLAG:-""}

echo "=== Training Config ==="
echo "MODEL=${MODEL} LR=${LR} BS=${BATCH_SIZE} EPOCHS=${EPOCHS}"
echo "DROPOUT=${DROPOUT} OPT=${OPTIMIZER} SCHED=${SCHEDULER}"
echo "RUN_NAME=${RUN_NAME}"
echo "======================="

python train.py \
    --model ${MODEL} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dropout ${DROPOUT} \
    --optimizer ${OPTIMIZER} \
    --scheduler ${SCHEDULER} \
    --num_conv_blocks ${NUM_CONV_BLOCKS} \
    --base_filters ${BASE_FILTERS} \
    --weight_decay ${WEIGHT_DECAY} \
    --run_name ${RUN_NAME} \
    --num_workers ${SLURM_CPUS_PER_TASK:-4} \
    ${PRETRAINED_FLAG}

conda deactivate
echo "job finished"

#!/bin/bash
#SBATCH -J crack_pred
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/predict_%j.txt
#SBATCH -e sbatchLog/crack_detection/predict_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /home/shi.bow/p-self/conda_env/crack_detection

echo "job start"

cd /home/shi.bow/p-self/course_project.INFO_6105/final

# ---- Required: set CHECKPOINT path ----
CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT, e.g. --export=ALL,CHECKPOINT=checkpoints/custom_cnn_lr0.001_bs64_best.pt"}
MODEL=${MODEL:-custom_cnn}
OUTPUT_CSV=${OUTPUT_CSV:-predictions.csv}

echo "=== Prediction ==="
echo "CHECKPOINT=${CHECKPOINT}"
echo "MODEL=${MODEL}"
echo "==================="

python predict.py \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --output_csv ${OUTPUT_CSV} \
    --num_workers ${SLURM_CPUS_PER_TASK:-4}

conda deactivate
echo "job finished"

#!/bin/bash
#SBATCH -J crack_all_analysis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/all_analysis_%j.txt
#SBATCH -e sbatchLog/crack_detection/all_analysis_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /home/shi.bow/p-self/conda_env/crack_detection

START_TIME=$(date +%s)
echo "job start: $(date)"

cd /home/shi.bow/p-self/course_project.INFO_6105/final

# ---- BV settings (override via --export) ----
BV_RUNS=${BV_RUNS:-10}
BV_TRAIN_RATIO=${BV_TRAIN_RATIO:-0.5}
BV_EPOCHS=${BV_EPOCHS:-10}
SKIP_BV=${SKIP_BV:-"--skip_bv"}   # default: only plot curves; set to "" to enable BV analysis

echo "=== All-checkpoint Analysis ==="
echo "BV_RUNS=${BV_RUNS}  BV_EPOCHS=${BV_EPOCHS}  BV_TRAIN_RATIO=${BV_TRAIN_RATIO}"
echo "================================"

python run_all_analysis.py \
    --checkpoints_dir checkpoints \
    --bv_runs ${BV_RUNS} \
    --bv_train_ratio ${BV_TRAIN_RATIO} \
    --bv_epochs ${BV_EPOCHS} \
    --num_workers ${SLURM_CPUS_PER_TASK:-4} \
    ${SKIP_BV}

conda deactivate
END_TIME=$(date +%s)
echo "job finished: $(date)"
echo "duration: $(( (END_TIME - START_TIME) / 3600 ))h $(( (END_TIME - START_TIME) % 3600 / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"

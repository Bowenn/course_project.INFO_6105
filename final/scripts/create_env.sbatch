#!/bin/bash
#SBATCH -J create_env
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/create_env_%j.txt
#SBATCH -e sbatchLog/crack_detection/create_env_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

START_TIME=$(date +%s)
echo "job start: $(date)"

conda env create -f /home/shi.bow/p-self/course_project.INFO_6105/final/environment.yml \
    -p /home/shi.bow/p-self/conda_env/crack_detection

echo "env created"

conda activate /home/shi.bow/p-self/conda_env/crack_detection
python --version
pip list

conda deactivate
END_TIME=$(date +%s)
echo "job finished: $(date)"
echo "duration: $(( (END_TIME - START_TIME) / 3600 ))h $(( (END_TIME - START_TIME) % 3600 / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"

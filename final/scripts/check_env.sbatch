#!/bin/bash
#SBATCH -J check_env
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH -D /home/shi.bow/sbatchDir
#SBATCH -o sbatchLog/crack_detection/check_env_%j.txt
#SBATCH -e sbatchLog/crack_detection/check_env_err_%j.txt
#SBATCH --mail-user=shi.bow@northeastern.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06
. /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /home/shi.bow/p-self/conda_env/crack_detection

START_TIME=$(date +%s)
echo "job start: $(date)"

echo "=== Python ==="
python --version

echo "=== PyTorch ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'cuDNN version:   {torch.backends.cudnn.version()}')
    print(f'GPU count:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
    # quick tensor test
    x = torch.randn(2, 3).cuda()
    print(f'Tensor on GPU:   {x.device} -- OK')
else:
    print('WARNING: CUDA is NOT available')
"

echo "=== Key Packages ==="
python -c "
import torchvision; print(f'torchvision:  {torchvision.__version__}')
import numpy;       print(f'numpy:        {numpy.__version__}')
import pandas;      print(f'pandas:       {pandas.__version__}')
import sklearn;     print(f'scikit-learn: {sklearn.__version__}')
import PIL;         print(f'Pillow:       {PIL.__version__}')
import matplotlib;  print(f'matplotlib:   {matplotlib.__version__}')
import mlxtend;     print(f'mlxtend:      {mlxtend.__version__}')
import tqdm;        print(f'tqdm:         {tqdm.__version__}')
import scipy;       print(f'scipy:        {scipy.__version__}')
"

conda deactivate
END_TIME=$(date +%s)
echo "job finished: $(date)"
echo "duration: $(( (END_TIME - START_TIME) / 3600 ))h $(( (END_TIME - START_TIME) % 3600 / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"
echo "=== All checks done ==="

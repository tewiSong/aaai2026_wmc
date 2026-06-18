#!/bin/bash
#SBATCH --job-name=nesydm_smoke
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/ibex/user/songt/.uv-cache
export UV_PROJECT_ENVIRONMENT=/ibex/user/songt/conda_envs/nesydm_venv
export WANDB_MODE=disabled
PY=/ibex/user/songt/conda_envs/nesydm_venv/bin/python
cd /ibex/user/songt/neurosymbolic-diffusion
$PY -c "import torch;print('CUDA available:',torch.cuda.is_available(),'| device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== running N=1 for 2 epochs ==="
$PY -u expressive/experiments/mnist_op/mnistop.py --N 1 --epochs 2 --test_every_epochs 1 --use_cuda true --use_wandb false 2>&1 | grep -E "Epoch time|\[TEST|cpu|cuda|NEW EPOCH|device" | head -30

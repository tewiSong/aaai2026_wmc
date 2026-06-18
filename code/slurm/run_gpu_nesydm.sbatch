#!/bin/bash
# Run the ACTUAL NeSyDM (neuro-symbolic diffusion, van Krieken 2025) on N-digit MNIST addition,
# N=1..4 -- the benchmark behind its published collapse curve. Real repo at
# /ibex/user/songt/neurosymbolic-diffusion (uv-synced venv). W&B disabled. The TestLogger is
# instrumented to echo accuracy_y/accuracy_w to stdout, so we read the per-epoch validation
# accuracy directly; the best validation accuracy_y over the run is the reported number.
#SBATCH --job-name=nesy_gpu_nesydm
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# NOTE: deliberately NO `set -e` -- the grep result-captures below return non-zero when a pattern
# is absent, which would abort the whole sweep. Each step guards its own failure instead.
echo "HOST=$(hostname) START"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/ibex/user/songt/.uv-cache
export UV_PROJECT_ENVIRONMENT=/ibex/user/songt/conda_envs/nesydm_venv
export WANDB_MODE=disabled
PY=/ibex/user/songt/conda_envs/nesydm_venv/bin/python
REPO=/ibex/user/songt/neurosymbolic-diffusion
cd "$REPO" || { echo "FATAL: cannot cd $REPO"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import torch;print('CUDA',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" || true

# Default config (batch 16, K=1024 samples, tuned lr/beta) -- the real method, unmodified.
# Endpoints N=1 and N=4 isolate whether the answer accuracy collapses as digits couple.
# Answer accuracy is logged as 'val/y_fTM', digit accuracy as 'val/w_acc_top'.
for N in 1 4; do
  echo "================ NeSyDM MNIST addition N=$N ================"
  LOG=/ibex/user/songt/nesydm_N$N.log
  $PY -u expressive/experiments/mnist_op/mnistop.py --N $N --epochs 20 \
      --test_every_epochs 1 --use_cuda true --use_wandb false > "$LOG" 2>&1 || true
  best_y=$(grep -oE "y_fTM=[0-9.]+" "$LOG" 2>/dev/null | sort -t= -k2 -g | tail -1 || true)
  best_w=$(grep -oE "w_acc_top=[0-9.]+" "$LOG" 2>/dev/null | sort -t= -k2 -g | tail -1 || true)
  echo "NeSyDM N=$N best answer-acc: $best_y | best digit-acc: $best_w"
done
echo "=== NeSyDM answer-accuracy (y_fTM) endpoints ==="
for N in 1 4; do
  b=$(grep -oE "y_fTM=[0-9.]+" /ibex/user/songt/nesydm_N$N.log 2>/dev/null | sort -t= -k2 -g | tail -1 || true)
  echo "N=$N: $b"
done

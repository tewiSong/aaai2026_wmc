#!/bin/bash
# #5: run the ACTUAL A-NeSI system on N-digit MNIST addition (N=1..4), the benchmark behind
# its published 0.97 -> 0.23 collapse. Real repo at /ibex/user/songt/a-nesi (pip-installed),
# W&B disabled. Reports the final validation accuracy per N.
#SBATCH --job-name=nesy_gpu_anesi
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

PY=/ibex/user/songt/conda_envs/nesy/bin/python
ANESI=/ibex/user/songt/a-nesi/anesi
cd "$ANESI"
set -e
export WANDB_MODE=disabled PYTHONPATH="$ANESI"
nvidia-smi || true

for N in 1 2 3 4; do
  echo "================ A-NeSI MNIST addition N=$N ================"
  $PY experiments/mnist_op/mnistop.py --config experiments/mnist_op/configs/n$N.yaml \
      2>&1 | tee /ibex/user/songt/anesi_N$N.log | grep -E "Val accuracy|NEW EPOCH" | tail -3
  acc=$(grep -oE "Val accuracy: [0-9.]+" /ibex/user/songt/anesi_N$N.log | tail -1)
  echo "A-NeSI N=$N final: $acc"
done
echo "=== A-NeSI collapse curve ==="
for N in 1 2 3 4; do echo "N=$N: $(grep -oE 'Val accuracy: [0-9.]+' /ibex/user/songt/anesi_N$N.log | tail -1)"; done

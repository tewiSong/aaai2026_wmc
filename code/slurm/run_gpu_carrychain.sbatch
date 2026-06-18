#!/bin/bash
# GPU experiment: the carry-chain stress test (Exp 3c). N-digit MNIST addition under the
# independence assumption; test accuracy collapses as N grows.
#SBATCH --job-name=nesy_gpu_carrychain
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

PY=/ibex/user/songt/conda_envs/nesy/bin/python
cd /home/songt/6a202176262820e495dbe48a/code/scripts
set -e

nvidia-smi || true
echo "=== Exp3c: carry-chain IA collapse (N=1..4, 3 seeds) ==="
$PY 03c_exp3_carrychain.py --Ns 1,2,3,4 --epochs 5 --seeds 0,1,2

echo "=== GPU carry-chain done ==="

#!/bin/bash
# Heavy CPU experiments: the core-cut trade-off curve and modular exact WMC for mf and bp.
# These repeatedly recompute min-fill treewidth on the 10k/26k-node graphs, so they get a
# long wall-clock limit and more memory.
#SBATCH --job-name=nesy_cpu_modular
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=batch
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

PY=/ibex/user/songt/conda_envs/nesy/bin/python
cd /home/songt/6a202176262820e495dbe48a/code/scripts
set -e

echo "=== Exp2: core-cut trade-off curve (mf, bp) ==="
$PY 02_exp2_modular_wmc.py --part corecut --namespaces mf,bp --target_tw 14

echo "=== Exp2: modular exact WMC over all modules (cc, mf, bp) ==="
$PY 02_exp2_modular_wmc.py --part modular --namespaces cc,mf,bp --target_tw 14

echo "=== CPU modular done ==="

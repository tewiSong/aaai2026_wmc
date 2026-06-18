#!/bin/bash
# Main CPU experiments on the EL++ normalized GO theory (NF1 + NF2).
#SBATCH --job-name=nesy_cpu_main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Use the conda env's interpreter directly (a complete environment); do not source
# ~/.bashrc, whose Ibex profile.d scripts return nonzero and would trip `set -e`.
PY=/ibex/user/songt/conda_envs/nesy/bin/python
cd /home/songt/6a202176262820e495dbe48a/code/scripts
set -e

echo "=== Exp0: dataset stats (theory cache prebuilt from the normalization) ==="
$PY 00_prepare_data.py

echo "=== Exp1: BP = exact on trees, error at reconvergences (cc) ==="
$PY 01_exp1_bp_reconvergence.py

echo "=== Exp2: treewidth (all namespaces) ==="
$PY 02_exp2_modular_wmc.py --part treewidth --namespaces cc,mf,bp

echo "=== Exp2: rank-r dial (cc cores) ==="
$PY 02_exp2_modular_wmc.py --part rankdial

echo "=== Task F: modular + rank-r separator, end-to-end vs gold ==="
$PY 06_modular_rankr_endtoend.py

echo "=== Cost-II: disjunctive (SROIQ) fragment ==="
$PY 04_disjunctive_fragment.py

echo "=== Exp3a: exact knowledge-compilation wall (SDD circuit size vs treewidth) ==="
$PY 03a_exp3_compile_wall.py

echo "=== Exp3b: top-k provenance drift (Scallop) ==="
$PY 03b_exp3_scallop_topk.py

echo "=== Exp2 (full theory): modular WMC with NF2 conjunctions ==="
$PY 07_modular_with_nf2.py

echo "=== Exp3d: real Scallop (scallopy) top-k on GO ==="
$PY 03d_real_scallop.py

echo "=== Exp summary ==="
$PY collect_results.py

echo "=== CPU main done ==="

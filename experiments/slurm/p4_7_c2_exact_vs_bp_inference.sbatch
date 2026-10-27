#!/bin/bash -l
#SBATCH --job-name=p4_7_c2_exact_vs_bp
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P4.7: C2 trained/predicted with exact marginals vs with loopy BP (BP given exact theta*), same Adam trainer.
CASES=("go_cc cnn" "go_cc fused" "go_mf cnn" "go_mf fused")
read -r O I <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="${O}_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p4_7_c2_exact_vs_bp_inference.py --ns "$O" --input "$I" --steps 300 --lr 0.05 --bootstrap 1000

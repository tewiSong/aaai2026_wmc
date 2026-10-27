#!/bin/bash -l
#SBATCH --job-name=p1_4b_bp_vs_gibbs_deepgoplus_bp
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.4b: GO bp vocabulary, BP vs monotone-bracketed Gibbs (50k sweeps x 32 chains), 200 random proteins.
RUN_TAG="part$SLURM_ARRAY_TASK_ID" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_4b_bp_vs_bracketed_gibbs_deepgoplus_bp.py \
    --part "$SLURM_ARRAY_TASK_ID" --parts 4 --proteins 200 --sweeps 50000 --burn-in 5000

#!/bin/bash -l
#SBATCH --job-name=p7_2b_fused_scores_go_bp
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P7.2b: DIAMOND gate (b) for GO bp on old test proteins, fused scores for the P7 GO bp targets.
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p7_2b_fused_scores_go_bp_with_diamond_gate.py

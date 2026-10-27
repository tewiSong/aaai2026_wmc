#!/bin/bash -l
#SBATCH --job-name=p5_2b_fused_scores_diamond_gate
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P5.2b: DIAMOND gate (b) on old test proteins, fused scores for P5 targets.
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p5_2b_fused_scores_with_diamond_gate.py

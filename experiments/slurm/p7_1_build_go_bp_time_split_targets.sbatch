#!/bin/bash -l
#SBATCH --job-name=p7_1_build_go_bp_time_split_targets
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P7.1: GO bp time-split no-knowledge targets, same rules as the locked P5 protocol (P5.1).
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p7_1_build_go_bp_time_split_targets.py

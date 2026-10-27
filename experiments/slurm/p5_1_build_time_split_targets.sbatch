#!/bin/bash -l
#SBATCH --job-name=p5_1_build_time_split_targets
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P5.1: time-split no-knowledge targets (Swiss-Prot 2021_04 vs 2024_01), labels per locked protocol.
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p5_1_build_time_split_targets.py

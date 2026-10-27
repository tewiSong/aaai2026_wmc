#!/bin/bash -l
#SBATCH --job-name=p1_0c_build_diamondscore
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P1.0c: realistic GO scores (DiamondScore on Swiss-Prot 2021_04, prop and raw variants).
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_0c_build_diamondscore_go_scores.py --threads 32

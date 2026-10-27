#!/bin/bash -l
#SBATCH --job-name=p4_7d_v2_logodds_bp_fidelity
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P4.7d v2: fidelity of log-odds BP at the exact theta* (100 proteins, GO cc / GO mf).
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p4_7d_v2_logodds_bp_at_exact_theta_star.py

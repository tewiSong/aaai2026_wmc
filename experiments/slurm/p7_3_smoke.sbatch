#!/bin/bash -l
#SBATCH --job-name=p7_3_smoke
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P7.3 smoke test (no lock, 60 old + 60 new proteins, separate cache and output files).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG=smoke_fused /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p7_3_go_bp_locked_time_split_evaluation.py --input fused --smoke 60 --workers 32 --bootstrap 20

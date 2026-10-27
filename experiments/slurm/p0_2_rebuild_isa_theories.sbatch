#!/bin/bash -l
#SBATCH --job-name=p0_2_rebuild_isa_theories
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P0.2: parse GO/UBERON/ChEBI, build is-a theories, compare structure stats with the paper.
/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p0_2_rebuild_isa_theories_and_check_paper_stats.py

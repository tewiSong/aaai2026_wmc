#!/bin/bash -l
#SBATCH --job-name=p0_4_determinism_profile
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P0.4: consistent-state counts of the min-fill bags (needs P0.3 min-fill orders).
ONTOS=(go_cc go_mf go_bp uberon chebi)
O=${ONTOS[$SLURM_ARRAY_TASK_ID]}
RUN_TAG="$O" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p0_4_determinism_profile_of_treewidth_bags.py \
    --ontology "$O" --heuristic min-fill

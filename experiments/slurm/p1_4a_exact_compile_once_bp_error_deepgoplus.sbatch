#!/bin/bash -l
#SBATCH --job-name=p1_4a_exact_compile_once_deepgoplus
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=224G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.4a: compile each DeepGOPlus vocabulary theory once with d4 (46 h cap), then exact
# marginals + BP error for all 3,874 held-out test proteins.
NS=(go_cc go_mf go_bp)
O=${NS[$SLURM_ARRAY_TASK_ID]}
RUN_TAG="$O" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_4a_exact_compile_once_bp_error_deepgoplus.py --ns "$O" --compile-timeout-hours 46

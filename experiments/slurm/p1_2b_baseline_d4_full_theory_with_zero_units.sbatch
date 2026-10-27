#!/bin/bash -l
#SBATCH --job-name=p1_2b_baseline_d4_zero_units
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.2b: plain d4 on the full theory with zero-score unit clauses, same proteins and limits as P1.2.
CASES=("go_bp prop" "go_bp raw" "go_mf prop")
read -r O V <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="${O}_${V}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_2b_baseline_d4_full_theory_with_zero_units.py \
    --ontology "$O" --diamond-key "$V" --d4-timeout 600 --verify 10

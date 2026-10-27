#!/bin/bash -l
#SBATCH --job-name=p1_2_certified_marginals_full_ontology
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.2: certified all-term marginals on the FULL ontology with real DiamondScore vectors
# (prune cold set, exact d4 on the active core), and BP's certified error against them.
CASES=("go_bp prop" "go_bp raw" "go_mf prop" "go_mf raw")
read -r O V <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="${O}_${V}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_2_certified_marginals_full_ontology_real_scores.py \
    --ontology "$O" --diamond-key "$V" --proteins 100 --clip 1e-9 --delta-target 1e-3 --d4-timeout 600

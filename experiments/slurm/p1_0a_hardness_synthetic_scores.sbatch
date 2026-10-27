#!/bin/bash -l
#SBATCH --job-name=p1_0a_hardness_synthetic_scores
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.0a: BP / Gibbs error against exact d4 marginals under synthetic score families,
# on full GO cc and ancestor-closed subgraphs (500/1000/2000 terms) of GO mf, GO bp, UBERON.
CASES=("go_cc 0" "go_mf 500" "go_mf 1000" "go_mf 2000" "go_bp 500" "go_bp 1000" "go_bp 2000" "uberon 500" "uberon 1000" "uberon 2000")
read -r O K <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="${O}_${K}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_0a_hardness_bp_gibbs_vs_exact_d4.py \
    --ontology "$O" --subgraph-size "$K" --n-subgraphs 4 --scores-per-family 2 \
    --gibbs-sweeps 3000 --d4-timeout 3600 --seed "$SLURM_ARRAY_TASK_ID"

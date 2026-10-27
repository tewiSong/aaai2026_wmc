#!/bin/bash -l
#SBATCH --job-name=p1_0a_hardness_real_diamond_scores
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.0a with real DiamondScore vectors (needs P1.0c). prop = hierarchy-consistent scores,
# raw = scores that can violate the hierarchy (like independent per-term classifiers).
CASES=("go_cc 0 prop" "go_cc 0 raw" "go_mf 1000 prop" "go_mf 1000 raw" "go_mf 2000 prop" "go_mf 2000 raw" "go_bp 1000 prop" "go_bp 1000 raw" "go_bp 2000 prop" "go_bp 2000 raw")
read -r O K V <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="${O}_${K}_${V}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_0a_hardness_bp_gibbs_vs_exact_d4.py \
    --ontology "$O" --subgraph-size "$K" --n-subgraphs 4 \
    --diamond /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/data/scores/diamondscore_$O.npz --diamond-key "$V" --proteins 12 \
    --gibbs-sweeps 3000 --d4-timeout 3600 --seed "$SLURM_ARRAY_TASK_ID" --tag "diamond_$V"

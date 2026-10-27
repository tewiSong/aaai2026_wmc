#!/bin/bash -l
#SBATCH --job-name=p0_3_minfill_width_check
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P0.3: whole-theory min-fill / min-degree width per ontology (10 array tasks).
ONTOS=(go_cc go_mf go_bp uberon chebi)
HEURS=(min-fill min-degree)
O=${ONTOS[$((SLURM_ARRAY_TASK_ID % 5))]}
H=${HEURS[$((SLURM_ARRAY_TASK_ID / 5))]}
RUN_TAG="${O}_${H}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p0_3_minfill_width_check_vs_paper.py \
    --ontology "$O" --heuristic "$H" --time-limit-hours 22

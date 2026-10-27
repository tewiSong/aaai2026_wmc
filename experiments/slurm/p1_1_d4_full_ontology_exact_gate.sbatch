#!/bin/bash -l
#SBATCH --job-name=p1_1_d4_full_ontology_exact_gate
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=224G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P1.1: gate test - d4 decision-DNNF compilation of each full ontology (46 h cap, 224 GB).
ONTOS=(go_mf go_bp uberon chebi)
O=${ONTOS[$SLURM_ARRAY_TASK_ID]}
RUN_TAG="$O" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p1_1_d4_compile_full_ontology_exact_gate.py --ontology "$O" --timeout-hours 46

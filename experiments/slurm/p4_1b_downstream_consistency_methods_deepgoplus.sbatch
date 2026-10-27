#!/bin/bash -l
#SBATCH --job-name=p4_1b_downstream_consistency_methods
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P4.1b: raw / max-closure / isotonic / exact WMC / BP vs true labels, standardised protocol.
CASES=("go_cc cnn" "go_cc fused" "go_mf cnn" "go_mf fused" "go_bp cnn" "go_bp fused")
read -r O I <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="${O}_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p4_1b_downstream_consistency_methods_deepgoplus.py --ns "$O" --input "$I" --bootstrap 1000

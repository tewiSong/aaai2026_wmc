#!/bin/bash -l
#SBATCH --job-name=p6_0_hot_core_regime
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P6.0: per-protein hot-core size / cold odds mass of the isotonic target, and d4 time on hot cores (GO bp vs cc/mf).
CASES=("go_bp cnn" "go_bp fused" "go_cc cnn" "go_cc fused" "go_mf cnn" "go_mf fused")
read -r O I <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="${O}_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p6_0_hot_core_regime_diagnostic.py --ns "$O" --input "$I"

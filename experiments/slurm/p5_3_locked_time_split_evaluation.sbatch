#!/bin/bash -l
#SBATCH --job-name=p5_3_locked_time_split_eval
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P5.3: locked time-split confirmation of C2 (fit on all old proteins, evaluate on 2024_01 no-knowledge targets).
CASES=("go_cc cnn" "go_cc fused" "go_mf cnn" "go_mf fused")
read -r O I <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="${O}_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p5_3_locked_time_split_evaluation.py --ns "$O" --input "$I" --workers 32 --bootstrap 1000

#!/bin/bash -l
#SBATCH --job-name=p7_3_go_bp_locked_eval
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P7.3: locked GO bp time-split confirmation (P7 protocol); core C2 + per-term affine (primary), C2, 9 baselines.
INPUTS=(fused cnn)
I=${INPUTS[$SLURM_ARRAY_TASK_ID]}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="go_bp_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p7_3_go_bp_locked_time_split_evaluation.py --input "$I" --workers 128 --bootstrap 1000

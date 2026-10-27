#!/bin/bash -l
#SBATCH --job-name=p7_4_go_bp_c2_diag
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P7.4: post-hoc diagnosis of the P7.3 C2 failure (time shift vs overfit, core vs cold, within vs across proteins).
INPUTS=(cnn fused)
I=${INPUTS[$SLURM_ARRAY_TASK_ID]}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="go_bp_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p7_4_go_bp_c2_time_shift_diagnosis.py --input "$I" --workers 128

#!/bin/bash -l
#SBATCH --job-name=p6_5_go_bp_tau_sens
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P6.5: GO bp core C2 with the P6.3 weights fixed, P6.3 core vs a strictly larger core (tau <= 0.007), 500 eval proteins.
INPUTS=(fused cnn)
I=${INPUTS[$SLURM_ARRAY_TASK_ID]}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="go_bp_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p6_5_go_bp_core_tau_sensitivity.py --input "$I" --sample 500 --workers 64 --bootstrap 1000

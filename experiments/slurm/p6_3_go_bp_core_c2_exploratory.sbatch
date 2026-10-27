#!/bin/bash -l
#SBATCH --job-name=p6_3_go_bp_core_c2
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=36:00:00
#SBATCH --array=0
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P6.3: C2 on GO bp with adaptive per-protein hot-core circuits vs isotonic and fitted non-WMC baselines (exploratory split).
INPUTS=(cnn fused)
I=${INPUTS[$SLURM_ARRAY_TASK_ID]}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="go_bp_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p6_3_go_bp_core_c2_exploratory.py --input "$I" --workers 64 --bootstrap 1000

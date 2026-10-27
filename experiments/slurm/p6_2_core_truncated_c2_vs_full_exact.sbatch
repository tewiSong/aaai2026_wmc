#!/bin/bash -l
#SBATCH --job-name=p6_2_core_truncated_c2
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P6.2: C2 on per-protein hot cores only (cold terms conditioned off) vs full exact C2, exploratory split.
CASES=("go_cc cnn" "go_cc fused" "go_mf cnn" "go_mf fused")
read -r O I <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_TAG="${O}_${I}" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p6_2_core_truncated_c2_vs_full_exact.py --ns "$O" --input "$I" --taus 0.1,0.03,0.01,0.003 --workers 32 --bootstrap 1000

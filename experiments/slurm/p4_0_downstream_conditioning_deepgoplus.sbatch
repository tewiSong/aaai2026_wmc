#!/bin/bash -l
#SBATCH --job-name=p4_0_downstream_conditioning_deepgoplus
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%A_%a.out
# P4.0: raw vs max-closure vs exact WMC vs BP against true labels (DeepGOPlus test proteins), bootstrap CIs.
NS=(go_cc go_mf)
O=${NS[$SLURM_ARRAY_TASK_ID]}
RUN_TAG="$O" /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p4_0_downstream_conditioning_deepgoplus_cc_mf.py --ns "$O" --bootstrap 1000

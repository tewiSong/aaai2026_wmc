#!/bin/bash -l
#SBATCH --job-name=p5_2a_deepgoplus_cnn_scores
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P5.2a: DeepGOPlus 1.0.6 CNN (TF 2.3.1 container) - gate (a) on old test proteins, then P5 targets.
LOG=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/p5_2a_deepgoplus_cnn_scores_with_reproducibility_gate_$(date +%Y%m%d-%H%M%S).log
{ echo "# script sha256 $(sha256sum /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/scripts/p5_2a_deepgoplus_cnn_scores_with_reproducibility_gate.py | cut -c1-64)"; echo "# sif $(sha256sum /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/env/tf/deepgoplus_tf.sif | cut -c1-16)"; echo "# job $SLURM_JOB_ID start $(date -Is)"; } > $LOG
set -o pipefail
cd /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments && /usr/bin/time -v singularity exec --bind /pfs,/scratch,/projappl,/project,/flash /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/env/tf/deepgoplus_tf.sif python -u scripts/p5_2a_deepgoplus_cnn_scores_with_reproducibility_gate.py 2>&1 | tee -a $LOG

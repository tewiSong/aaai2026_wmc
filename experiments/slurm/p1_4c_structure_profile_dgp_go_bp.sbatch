#!/bin/bash -l
#SBATCH --job-name=p1_4c_structure_profile_dgp_go_bp
#SBATCH --account=project_462001792
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/logs/slurm/%x-%j.out
# P1.4c: min-fill width and determinism profile of the DeepGOPlus GO bp vocabulary theory (4,474 terms).
RUN_TAG=dgp_go_bp /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p0_3_minfill_width_check_vs_paper.py --ontology dgp_go_bp --heuristic min-fill
RUN_TAG=dgp_go_bp /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/experiments/run.sh scripts/p0_4_determinism_profile_of_treewidth_bags.py --ontology dgp_go_bp --heuristic min-fill

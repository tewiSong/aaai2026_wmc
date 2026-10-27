#!/bin/bash -l
# Build the project container experiments/env/wmc_env.sif with cotainr (LUMI-recommended:
# one .sif file on Lustre instead of a conda tree with ~100k small files).
# Run on a login node; takes roughly 10-30 minutes.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ENV_DIR/../logs"
STAMP="$(date +%Y%m%d-%H%M%S)"

module load CrayEnv
module load cotainr/2025.7.1

# Build sandbox on node-local /tmp, never in $HOME (20 GB quota).
export TMPDIR="/tmp/$USER/cotainr_build_$STAMP"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

cd "$LOG_DIR"
cotainr build "$ENV_DIR/wmc_env.sif" \
    --system=lumi-c \
    --conda-env="$ENV_DIR/wmc_env.yml" \
    --accept-licenses \
    2>&1 | tee "$LOG_DIR/env_build_wmc_container_$STAMP.log"

sha256sum "$ENV_DIR/wmc_env.yml" "$ENV_DIR/wmc_env.sif" | tee -a "$LOG_DIR/env_build_wmc_container_$STAMP.log"
"$ENV_DIR/run_in_wmc_env.sh" python -c \
    "import numpy, scipy, pandas, networkx, cotengra, pysdd; print('container OK: numpy', numpy.__version__)" \
    2>&1 | tee -a "$LOG_DIR/env_build_wmc_container_$STAMP.log"

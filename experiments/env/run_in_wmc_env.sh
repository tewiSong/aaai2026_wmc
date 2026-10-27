#!/bin/bash
# Run a command in the project environment.
#   - If env/wmc_env.sif exists (LUMI: cotainr container), run inside it, binding the host file
#     systems that exist on this machine.
#   - Otherwise run directly; the conda env from env/wmc_env.yml must be active
#     (`conda activate wmc`). This is the path on servers without the container.
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$ENV_DIR/wmc_env.sif"
if [[ -f "$SIF" ]]; then
    BINDS=""
    for d in /pfs /scratch /projappl /project /flash /appl; do
        [[ -d "$d" ]] && BINDS="${BINDS:+$BINDS,}$d"
    done
    EXP_REAL="$(cd "$ENV_DIR/.." && pwd -P)"
    DATA_REAL="$(readlink -f "$ENV_DIR/../data" 2>/dev/null || true)"
    for d in "$EXP_REAL" "$DATA_REAL"; do
        [[ -n "$d" && -d "$d" ]] && BINDS="${BINDS:+$BINDS,}$d"
    done
    exec singularity exec ${BINDS:+--bind "$BINDS"} "$SIF" "$@"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "wmc" ]]; then
    echo "warning: no container at $SIF and conda env 'wmc' is not active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none})" >&2
fi
exec "$@"

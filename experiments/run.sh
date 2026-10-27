#!/bin/bash
# Run one experiment script inside the container and keep a provenance log.
#   experiments/run.sh scripts/<name>.py [args...]
#   experiments/run.sh scripts/<name>.sh [args...]
# Log: experiments/logs/<name>[_<tag>]_<timestamp>.log  (set RUN_TAG=... to add a tag)
set -uo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$1"; shift
[[ "$SCRIPT" = /* ]] || SCRIPT="$EXP_DIR/$SCRIPT"
if [[ ! -f "$SCRIPT" ]]; then echo "no such script: $SCRIPT" >&2; exit 1; fi

NAME="$(basename "${SCRIPT%.*}")"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$EXP_DIR/logs/${NAME}${RUN_TAG:+_$RUN_TAG}_${STAMP}.log"

case "$SCRIPT" in
    *.py) CMD=(python3 -u "$SCRIPT" "$@") ;;
    *.sh) CMD=(bash "$SCRIPT" "$@") ;;
    *) echo "unsupported script type: $SCRIPT" >&2; exit 1 ;;
esac

{
    echo "# script : $SCRIPT"
    echo "# sha256 : $(sha256sum "$SCRIPT" | cut -d' ' -f1)"
    echo "# args   : $*"
    echo "# host   : $(hostname)   slurm_job: ${SLURM_JOB_ID:-none}"
    echo "# start  : $(date -Is)"
    echo "# sif    : $(sha256sum "$EXP_DIR/env/wmc_env.sif" 2>/dev/null | cut -c1-16)"
    echo
} > "$LOG"

cd "$EXP_DIR"
/usr/bin/time -v "$EXP_DIR/env/run_in_wmc_env.sh" "${CMD[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo -e "\n# end    : $(date -Is)   exit: $RC" >> "$LOG"
echo "log written to $LOG"
exit "$RC"

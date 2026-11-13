#!/usr/bin/env bash
# One task of stage 4: the dump directories of a list whose line number is TASK modulo
# NTASKS, aligned one after the other on this task's GPU (local/v2/align.py; the
# remaining arguments go to it).
#   local/v2/align_task.sh <dump list> <task (1-based)> <ntasks> [align.py options]
set -euo pipefail
list=$1; task=$2; ntasks=$3; shift 3
v2=$(dirname "$0")
awk -v t="${task}" -v n="${ntasks}" '(NR - 1) % n == t - 1' "${list}" | while read -r d; do
    [ -f "${d}/.aligned" ] && continue
    echo "$(date '+%F %T') ${d}"
    python3 "${v2}/align.py" --dump "${d}" --device cuda "$@"
done

#!/usr/bin/env bash
# Run line TASK of the command list LIST (one task of a ${train_cmd} JOB=1:N array).
set -euo pipefail
cmd=$(sed -n "${2}p" "$1")
echo "${cmd}"
eval "${cmd}"

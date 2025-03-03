#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


./run_entailment.sh "$@" &
sleep 2s

./run_aqa_yn.sh "$@" &
sleep 2s

./run_aqa_open.sh "$@" &

wait
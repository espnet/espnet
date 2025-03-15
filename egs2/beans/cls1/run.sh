#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

## Detection

./run_dcase.sh "$@" &
sleep 2s

./run_enabirds.sh "$@" &
sleep 2s

./run_gibbons.sh "$@" &
sleep 2s

./run_hiceas.sh "$@" &
sleep 2s

./run_rfcx.sh "$@" &
sleep 2s

## Classification

# ./run_watkins.sh "$@" &
# sleep 2s

# ./run_bats.sh "$@" &
# sleep 2s

# ./run_cbi.sh "$@" &
# sleep 2s

# ./run_humbugdb.sh "$@" &
# sleep 2s

# ./run_dogs.sh "$@" &

wait

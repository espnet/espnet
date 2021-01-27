#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=data/train_set
valid_set=data/valid_set
test_sets=data/test_set

./diar.sh \
    --stage 1 \
    --stop_stage 1 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lang en \
    --ngpu 1 \
    --local_data_opts "" \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test "

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lang en \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_sgmse_ncsnpp.yaml \
    "$@"

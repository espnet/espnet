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
    --ngpu 1 \
    --ref_num 1 \
    --nj 8 \
    --enh_config conf/tuning/train_enh_blstm_tf.yaml \
    "$@"

    # --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \

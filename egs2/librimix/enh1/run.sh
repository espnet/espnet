#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k
min_or_max=min  # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.


train_set="train"
valid_set="dev"
test_sets="test "

./enh.sh \
    --use_noise_ref true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --audio_format wav \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --lang en \
    --ngpu 4 \
    --enh_config ./conf/train.yaml \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

num_train=20000
num_dev=5000
num_eval=3000
sample_rate=16k

# 0, 1, 2, 3 represent drums, bass, vocals, and others, respectively.
ref_num=4


train_set="train_${sample_rate}"
valid_set="dev_${sample_rate}"
test_sets="test_${sample_rate} "

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --audio_format wav \
    --ref_num ${ref_num} \
    --lang en \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --num_train ${num_train} --num_dev ${num_dev} --num_eval ${num_eval}" \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    "$@"

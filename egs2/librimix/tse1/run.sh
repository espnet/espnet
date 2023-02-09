#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k # If using 8k, please make sure `spk2enroll.json` points to 8k audios as well
min_or_max=min  # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.


train_set="train"
valid_set="dev"
test_sets="test "

./enh.sh \
    --is_tse_task true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --lang en \
    --ngpu 4 \
    --enh_config ./conf/train.yaml \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=44.1k
ch=1

train_set="train_${fs}_${ch}ch"
valid_set="dev_${fs}_${ch}ch"
test_sets="test_${fs}_${ch}ch"

train_config=conf/train.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml

./codec.sh \
    --fs ${fs} \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${fs} --nchannels ${ch}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --audio_format wav \
    "$@"

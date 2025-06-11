#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16k

train_set="train"
valid_set="dev"

#Set test_dets="test_stubset" to use a subset of the test set (1000 files) for evaluation
test_sets="test"

use_unbal=false
dev_ratio=0.02

train_config=conf/train.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml

./codec.sh \
    --fs ${fs} \
    --ngpu 1 \
    --audio_format wav \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "--use_unbal ${use_unbal} --dev_ratio ${dev_ratio}"
    "$@"

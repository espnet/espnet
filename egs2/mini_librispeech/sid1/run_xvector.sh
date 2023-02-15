#!/usr/bin/env bash

# Copyright 2023 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=test

train_config=conf/train_sid_transformer_xvector.yaml
decode_config=conf/decode_sid.yaml

./asr.sh \
    --stage 10 \
    --feats_type extracted \
    --feats_normalize utterance_mvn \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --asr_config "${train_config}" \
    --use_lm false \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --token_type word \
    --local_data_opts "--stage 0" \
    "$@"

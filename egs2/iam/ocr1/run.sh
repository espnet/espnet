#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_set="test"

asr_config=conf/train_asr_conformer.yaml

./asr.sh \
    --lang en \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_set} \
    --feats_type extracted \
    --use_lm false \
    --token_type char \
    --asr_config ${asr_config} \
    --ngpu 4 \
    --gpu_inference true \

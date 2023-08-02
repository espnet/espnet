#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

lm_train_text=data/lm_train.txt
lm_dev_text=data/lm_valid.txt
lm_test_text=data/lm_test.txt

lm_config=conf/tuning/train_lm_rnn_unit1024_nlayers3_dropout0.2_epoch30.yaml

./asr.sh \
    --stage 6 \
    --stop_stage 8 \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 1 \
    --gpu_inference true \
    --train_set "dummy_train" \
    --valid_set "dummy_valid" \
    --test_sets "dummy_test" \
    --token_type word \
    --lm_config "${lm_config}" \
    --lm_train_text "${lm_train_text}" \
    --lm_dev_text "${lm_dev_text}" \
    --lm_test_text "${lm_test_text}" "$@"

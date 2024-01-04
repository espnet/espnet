#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"
local_data_opts="--use_transcript false"

slu_config=conf/train_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --speed_perturb_factors '0.9 1.0 1.1'\
    --inference_nj 8 \
    --inference_slu_model valid.acc.ave_10best.pth\
    --slu_config "${slu_config}" \
    --lm_train_text dump/raw/train_sp/text \
    --lm_train_transcript dump/raw/train_sp/transcript \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --local_data_opts "${local_data_opts}" \
    --test_sets "${test_sets}" "$@"

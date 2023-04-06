#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

slu_config=conf/tuning/train_asr_no_pretrain.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --feats_type raw\
    --stage 15 \
    --skip_upload_hf false \
    --hf_repo espnet/siddhana_slue_asr_train_asr_conformer_raw_en_word_valid.acc.ave_10best \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_slu_model valid.acc.ave_10best.pth\
    --slu_config "${slu_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

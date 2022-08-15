#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Add path of first pass model as pretrained_model
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

asr_config=conf/tuning/train_asr_bert_conformer_deliberation.yaml

./slu_transcript.sh \
    --lang en \
    --ngpu 1 \
    --stage 11\
    --use_transcript true \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 12 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

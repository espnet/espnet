#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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
    --use_transcript true \
    --use_lm false \
    --nbpe 5000 \
    --stage 1 \
    --stop_stage 1 \
    --pretrained_model ../../slurp_new/asr1/exp/asr_train_asr_conformer_raw_en_word/valid.acc.ave_10best.pth:encoder:encoder\
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

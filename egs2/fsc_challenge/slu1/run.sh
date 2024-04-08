#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Get first pass model from here -  https://huggingface.co/espnet/siddhana_fsc_challenge_asr_train_asr_hubert_transformer_adam_specaug_raw_en_word_valid.acc.ave_5best to initialise encoder
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="utt_test spk_test valid"

gt=false

if ${gt}; then
    local_data_opts="--gt true"
    slu_config=conf/tuning/train_asr_hubert_transformer_adam_specaug_deliberation_transformer_gt.yaml
else
    local_data_opts="--gt false"
    slu_config=conf/train_asr.yaml
fi

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --stage 1\
    --stop_stage 1\
    --nj 1\
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 30 \
    --use_transcript true\
    --pretrained_model ../../fsc_challenge/asr1/exp/asr_train_asr_hubert_transformer_adam_specaug_old_raw_en_word/valid.acc.ave_5best.pth:encoder:encoder\
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_slu_model valid.acc.ave_5best.pth\
    --slu_config "${slu_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "${local_data_opts}" "$@"

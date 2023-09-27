#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_combined"
valid_set="valid"
test_sets="test_snips"

asr_config=conf/train_asr_whisper_full_correct_specaug.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --use_prompt true \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --nlsyms_txt add_tokens-Copy1.txt \
    --inference_nj 8 \
    --audio_format "flac.ark" \
    --inference_asr_model valid.acc.ave.pth\
    --inference_config conf/decode_asr_ic.yaml\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

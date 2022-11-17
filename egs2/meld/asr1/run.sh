#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="valid test"

asr_config=conf/tuning/train_asr_hubert_transformer_adam_specaug_meld.yaml
inference_config=conf/decoder_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --stop_stage 13 \
    --inference_config "${inference_config}" \
    --use_lm false \
    --nbpe 850 \
    --token_type bpe\
    --bpe_train_text "dump/raw/train/text" \
    --bpe_nlsyms sadness,surprise,neutral,joy,anger,fear,disgust\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 20 \
    --feats_normalize utterance_mvn\
    --inference_nj 4 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



test_set="test test_sub"
train_set="dev_train"
valid_set="dev_non_train"
nbpe=500 
asr_config=conf/train_asr_wavlm_transformer.yaml
inference_config=conf/decode_asr.yaml


./asr.sh \
    --use_lm false \
    --lang en \
    --ngpu 1 \
    --nbpe "${nbpe}" \
    --max_wav_duration 20 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "wav" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --feats_normalize uttmvn \
    --bpe_train_text "data/${train_set}/text" "$@"

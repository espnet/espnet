#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

slu_config=conf//train_asr_wavlm.yaml
inference_config=conf/decode_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type bpe \
    --gpu_inference true \
    --nbpe 1000 \
    --bpe_nlsyms ANS,SEP \
    --feats_type raw \
    --audio_format "flac.ark" \
    --max_wav_duration 120 \
    --feats_normalize utterance_mvn \
    --slu_config "${slu_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text dump/raw/train/text \
    --bpe_train_text "data/${train_set}/text" "$@"

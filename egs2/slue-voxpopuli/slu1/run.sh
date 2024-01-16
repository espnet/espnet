#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

slu_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type bpe \
    --gpu_inference true \
    --nbpe 1000 \
    --bpe_nlsyms FILL,SEP,PLACE,QUANT,ORG,WHEN,NORP,PERSON,LAW \
    --feats_type raw \
    --audio_format "flac.ark" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn \
    --slu_config "${slu_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text dump/raw/train_sp/text \
    --lm_train_transcript dump/raw/train_sp/transcript \
    --bpe_train_text "data/${train_set}/text" "$@"

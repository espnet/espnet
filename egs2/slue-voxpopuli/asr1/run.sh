#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

asr_config=conf/tuning_wavlm/train_asr_conformer_lr2e-3_warmup5k_wavlm_conv2d2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type bpe \
    --nbpe 1000 \
    --bpe_nlsyms FILL,SEP,PLACE,QUANT,ORG,WHEN,NORP,PERSON,LAW \
    --feats_type raw \
    --audio_format "flac.ark" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

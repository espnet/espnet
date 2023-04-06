#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="train_sp"

slu_config=conf/tuning_wavlm/train_asr_conformer_lr5e-4_warmup5k_conv2d.yaml
inference_config=conf/decode_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type bpe \
    --stage 12 \
    --stop_stage 13 \
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
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --lang ko \
    --asr_config conf/train_asr_conformer.yaml \
    --lm_config conf/train_lm_transformer.yaml \
    --inference_config conf/decode_asr.yaml \
    --train_set train \
    --valid_set val \
    --test_sets "test" \
    --nbpe 2000 \
    --gpu_inference true \
    --lm_train_text "data/train/text" "$@" \

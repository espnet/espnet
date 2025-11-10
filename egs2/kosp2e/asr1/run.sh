#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --nj 8 \
    --inference_nj 8 \
    --lang ko \
    --asr_config conf/train_asr_conformer.yaml \
    --lm_config conf/train_lm_transformer.yaml \
    --inference_config conf/decode_asr.yaml \
    --stage 2 \
    --train_set train \
    --valid_set val \
    --test_sets "test" \
    --nbpe 2000 \
    --ngpu 4 \
    --batch_size 1 \
    --gpu_inference true \
    --lm_train_text "data/train/text" "$@" \

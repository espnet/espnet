#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr2.sh \
    --nj 2 \
    --inference_nj 2 \
    --kmeans_feature "mfcc" \
    --nclusters "10" \
    --use_lm false \
    --src_lang "mfcc_km10" \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --asr_config conf/train_asr_transformer_debug.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --lm_train_text "data/train_nodev/text.ts.en" "$@"

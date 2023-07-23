#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./st2.sh \
    --nj 2 \
    --inference_nj 2 \
    --kmeans_feature "mfcc" \
    --nclusters "10" \
    --kmeans_opts "--nj 1" \
    --use_lm false \
    --speech_token_lang "mfcc_km10" \
    --src_tgt_text_lang "en/en" \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --st_config conf/train_asr_transformer.yaml \
    --tgt_tasks "asr/mt/st" \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --src_bpe_train_text "data/train_nodev.asr_mt_st/text.ts.mfcc_km10" \
    --tgt_bpe_train_text "data/train_nodev.asr_mt_st/text.lc.rm.en_en" \
    --lm_train_text "data/train_nodev/text.lc.rm.en" "$@"

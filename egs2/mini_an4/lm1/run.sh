#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lm_inference_asr_config=conf/decode_lm_asr.yaml
lm_inference_tts_config=conf/decode_lm_tts.yaml

./lm.sh \
    --kmeans_feature "mfcc" \
    --learn_kmeans true \
    --nclusters "10" \
    --num_splits_lm 1 \
    --lang "mfcc_km10" \
    --token_type char \
    --nlsyms_txt data/nlsyms.txt \
    --lm_config conf/train_transformer.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --lm_inference_asr_config "${lm_inference_asr_config}" \
    --lm_inference_tts_config "${lm_inference_tts_config}" \
    --lm_test_text_asr dump/raw/test/text.asr \
    --lm_test_text_tts dump/raw/test/text.tts "$@"

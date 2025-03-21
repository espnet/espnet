#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --lang en \
    --asr_config conf/train_asr_s3prl_single_lr1e-3.yaml \
    --inference_config conf/decode_asr_ctc.yaml \
    --lm_config conf/train_lm.yaml \
    --feats_normalize utterance_mvn \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --gpu_inference true \
    --inference_nj 1 \
    --train_set train \
    --valid_set dev \
    --test_sets "test/Zacatlan test/Tequila test/Hidalgo" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"

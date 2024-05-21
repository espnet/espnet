#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :

# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e

set -u

set -o pipefail

./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --nj 4 \
    --ngpu 1 \
    --nbpe 300 \
    --gpu_inference true \
    --inference_nj 1 \
    --use_lm false \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --lang en \
    --token_type bpe \
    --asr_config conf/train_asr_demo_branchformer.yaml \
    --inference_args "--beam_size 10 --ctc_weight 0.3" \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --bpe_train_text "data/train_nodev/text" \
    --lm_train_text "data/train_nodev/text" "$@"

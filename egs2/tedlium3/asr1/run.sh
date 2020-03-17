#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config="conf/transformer_config.yaml"

./asr.sh \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --nbpe 500 \
    --train_set train \
    --dev_set dev \
    --eval_sets "test" \
    --feats_type fbank_pitch \
    --lm_train_text "data/local/lm_train/train.txt" \
    --srctexts dump/fbank_pitch/train_sp/text \
    --nbpe 500 \
    --asr_config ${config} \
    --use_word_lm false \
    --decode_asr_model valid.loss.ave.pth \
    "$@"

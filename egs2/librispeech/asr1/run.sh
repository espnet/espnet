#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config="conf/transformer_config.yaml"

decode_asr_model=valid.loss.ave.pth # valid.acc.best.pth

./asr.sh \
    --nbpe 5000 \
    --train_set train_960 \
    --dev_set dev \
    --eval_sets "test_clean test_other dev_clean dev_other " \
    --srctexts "dump/fbank_pitch/train_960/text" \
    --feats_type fbank_pitch \
    --lm_train_text "data/local/lm_train/train.txt" \
    --lm_dev_text "data/local/lm_train/dev.txt" \
    --asr_config ${config} \
    --use_word_lm false \
    --decode_asr_model ${decode_asr_model} \
    "$@"

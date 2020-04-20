#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config="conf/train_asr_transformer.yaml"

decode_asr_model=valid.loss.ave.pth # valid.acc.best.pth

./asr.sh \
    --stage 0 --stop_stage 13 \
    --nbpe 5000 \
    --train_set train_960 \
    --dev_set dev \
    --eval_sets "test_clean test_other dev_clean dev_other " \
    --srctexts "dump/raw/train_960/text" \
    --feats_type raw \
    --lm_train_text "data/local/lm_train/train.txt" \
    --asr_config ${config} \
    --use_word_lm false \
    --decode_asr_model ${decode_asr_model} \
    "$@"

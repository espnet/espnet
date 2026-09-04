#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960_sp"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_spiralformer_finetune.yaml
inference_config=conf/tuning/decode_asr_ctc.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --feats_type raw \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_args "--init_param /pretrained/model.pth:encoder:encoder /pretrained/model.pth:ctc:ctc" \
    --ignore_init_mismatch true \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --use_lm false \
    --bpe_train_text "data/${train_set}/text" "$@"

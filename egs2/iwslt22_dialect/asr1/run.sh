#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
test_set=test1

asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr.yaml

nbpe=1000

./asr.sh \
    --ngpu 1 \
    --stage 1 \
    --stop_stage 13 \
    --audio_format "flac.ark" \
    --local_data_opts "--stage 0" \
    --use_lm false \
    --token_type bpe \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --inference_nj 40 \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"

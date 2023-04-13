#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
test_sets="dev"

s2t_config=conf/tuning/train_s2t_conformer.yaml
inference_config=conf/tuning/decode_s2t_conformer.yaml

nbpe=10000

./s2t.sh \
    --use_lm false \
    --ngpu 4 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 4 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"

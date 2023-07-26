#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev_seen"
test_sets="dev_seen test_seen test_unseen"

aai_config=conf/train_aai.yaml
inference_config=conf/decode_aai.yaml

./aai.sh \
    --lang en \
    --stage 5 \
    --ngpu 1 \
    --nj 16 \
    --feats_type raw \
    --max_wav_duration 30 \
    --aai_config "${aai_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_aai_model "10epoch.pth" \

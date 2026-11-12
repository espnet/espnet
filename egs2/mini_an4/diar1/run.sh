#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./diar.sh \
    --nj 2 \
    --inference_nj 2 \
    --fs 16k \
    --num_spk 2 \
    --collar 0.0 \
    --train_set train \
    --valid_set test \
    --test_sets test \
    --diar_config ./conf/train_diar_debug.yaml \
    --inference_config ./conf/decode_diar.yaml \
    "$@"

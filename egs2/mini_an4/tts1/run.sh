#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./tts.sh \
    --nj 2 \
    --inference_nj 2 \
    --lang en \
    --train_config conf/train_tacotron2_debug.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --srctexts "data/train_nodev/text" "$@"

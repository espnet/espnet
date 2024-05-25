#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# TODO(jiatong): to add train config without prior info (e.g., tacotron-based etc)

./tts.sh \
    --nj 2 \
    --inference_nj 2 \
    --lang en \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --srctexts "data/train_nodev/text" "$@"

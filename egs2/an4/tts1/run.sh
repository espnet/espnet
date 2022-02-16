#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./tts.sh \
    --lang en \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --srctexts "data/train_nodev/text" "$@"

#!/usr/bin/env bash
set -e
set -u
set -o pipefail

./tts.sh \
    --lang rw \
    --token_type char \
    --cleaner none \
    --g2p none \
    --train_set "train" \
    --valid_set "dev" \
    --test_sets "dev test" \
    --train_config "conf/train.yaml" \
    --inference_config "conf/decode.yaml" \
    "$@"

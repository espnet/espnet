#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./speechlm.sh \
    --task "plain_tts" \
    --codec_token_per_frame 12 \
    --nj 1 \
    --inference_nj 2 \
    --audio_format "flac.ark" \
    --train_config conf/train.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "test" \
    "$@"

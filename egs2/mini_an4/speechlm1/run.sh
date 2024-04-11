#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# format: file,name,type

./speechlm.sh \
    --stage 7 --stop_stage 8 \
    --task "plain_tts" \
    --codec_token_per_frame 12 \
    --codec_token_in_use 12 \
    --nj 1 \
    --inference_nj 2 \
    --audio_format "flac.ark" \
    --train_config conf/train.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "test" \
    "$@"

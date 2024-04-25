#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train-clean-460
valid_set=dev-clean
test_sets="dev-clean test-clean"

train_config=conf/train_multiscale.yaml
inference_config=conf/decode.yaml

./speechlm.sh \
    --stage 8 --stop_stage 8 \
    --task "tts" \
    --ngpu 1 \
    --nj 32 \
    --inference_nj 2 \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"

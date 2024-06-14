#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=44100

train_set=train
valid_set=dev-small
test_sets="speech_test audio_test music_test"

train_config=conf/train_soundstream4_fs44100.yaml
inference_config=conf/decode.yaml

./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --nj 60 \
    --gpu_inference true \
    --audio_format flack.ark \
    --inference_nj 2 \
    --fs ${fs} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

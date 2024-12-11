#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

train_set=train-clean-100
valid_set=dev-clean
test_sets=test-clean

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 1 \
    --ref_channel 1 \
    --enh_config conf/train.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --audio_format flac \
    "$@"

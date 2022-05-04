#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

train_set=train_multich 
valid_set=dev_multich
test_sets=test_multich


./enh.sh --audio_format wav \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --spk_num 1 \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.snr.best.pth" \
    "$@"

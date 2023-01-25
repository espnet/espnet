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

# train_set=train_singlech
# valid_set=dev_singlech
# test_sets=test_singlech

./enh.sh --audio_format wav \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_dprnntac_fasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.ave.pth" \
    "$@"

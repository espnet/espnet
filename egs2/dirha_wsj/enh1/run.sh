#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# see lines 23-24 in `local/data.sh` for more information
ref_mic=Beam_Circular_Array

train_set="train_si284_${ref_mic}"
valid_set="dirha_sim_${ref_mic}"
test_sets="dirha_real_${ref_mic}"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs 16k \
    --ngpu 2 \
    --spk_num 1 \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k


train_set=tr05_simu
valid_set=dt05_simu
test_sets="et05_simu"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --spk_num 1 \
    --local_data_opts "--sample_rate ${sample_rate}" \
    --enh_config ./conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    "$@"

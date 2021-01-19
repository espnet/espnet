#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k


# train_set=tr05_simu_isolated_6ch_track
# valid_set=dt05_simu_isolated_6ch_track
# test_sets="et05_simuz_isolated_6ch_track"

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets="et05_simu_isolated_1ch_track"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --spk_num 1 \
    --local_data_opts "--sample_rate ${sample_rate}" \
    --enh_config ./conf/tuning/train_enh_PSM.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

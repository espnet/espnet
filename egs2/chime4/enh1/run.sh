#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=

# train_set=tr05_simu_isolated_6ch_track
# valid_set=dt05_simu_isolated_6ch_track
# test_sets="et05_simu_isolated_6ch_track"

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets="et05_simu_isolated_1ch_track"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --ref_num 1 \
    --ref_channel 3 \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 1 --stop-stage 2" \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

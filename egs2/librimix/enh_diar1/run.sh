#!/usr/bin/env bash

# Copyright 2022 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Note: To train the model with a flexible number of speakers,
# run local/run_adapt.sh after running this script until stage 5 (training).

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

train_config="conf/tuning/train_diar_enh_convtasnet_2.yaml"
decode_config="conf/tuning/decode_diar_enh.yaml"
num_spk=2 # 2, 3

./enh_diar.sh \
    --use_noise_ref true \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 32 \
    --audio_format wav \
    --local_data_opts "--num_spk ${num_spk}" \
    --ref_num "${num_spk}"\
    --frame_shift 64 \
    "$@"

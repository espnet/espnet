#!/usr/bin/env bash

# Copyright 2022 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Note: This script is for adapting the model to a flexible number of speakers using 2- & 3-speaker dataset
# using the pre-trained model trained on 2-speaker dataset.

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

train_config="conf/tuning/train_diar_enh_convtasnet_adapt.yaml"
decode_config="conf/tuning/decode_diar_enh_adapt.yaml"
# change the path according to the actual path to the pretrained model
pretrained="exp/diar_enh_train_diar_enh_convtasnet_2_raw/valid.loss_enh.best.pth"
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
    --local_data_opts "--stage 2 --adapt True" \
    --diar_args "--init_param ${pretrained}" \
    --diar_tag "train_diar_enh_convtasnet_adapt" \
    --spk_num "3"\
    --frame_shift 64 \
    "$@"

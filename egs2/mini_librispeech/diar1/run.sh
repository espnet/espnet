#!/bin/bash

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set=simu/data/train_clean_5_ns2_beta2_500
valid_set=simu/data/dev_clean_2_ns2_beta2_500
test_sets=simu/data/dev_clean_2_ns2_beta2_500

train_config=conf/train.yaml

export CUDA_VISIBLE_DEVICES=0
./diar.sh \
    --stage 5 \
    --stop_stage 5 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --local_data_opts "--stage 1" \
    "$@"

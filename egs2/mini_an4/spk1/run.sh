#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_mini_RawNet3.yaml

./spk.sh \
    --spk_config ${spk_config} \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    "$@"

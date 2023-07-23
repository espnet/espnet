#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_RawNet3.yaml

train_set="voxceleb12_devs"
valid_set="voxceleb1_test"
test_sets="voxceleb1_test"

./spk.sh \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@"

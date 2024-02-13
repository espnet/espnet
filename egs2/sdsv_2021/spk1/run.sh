#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_rawnet3.yaml

train_set="combined_train_set"
valid_set="voxceleb1_test"
test_sets="SampleDeepMine"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@"

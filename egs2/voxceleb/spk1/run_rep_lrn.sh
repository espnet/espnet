#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/tuning/train_rawnet3_decoder_spkonly.yaml

train_set="voxceleb12_devs_sp"
valid_set="voxceleb1_test"
test_sets="voxceleb1_test"
feats_type="raw"

./replrn.sh \
    --spk_config ${spk_config} \
    --feats_type ${feats_type} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    --stage 5 \
    "$@"

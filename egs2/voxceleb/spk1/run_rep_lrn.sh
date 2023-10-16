#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/tuning/train_rawnet3_decoder_spkonly.yaml

#train_set="voxceleb12_devs_sp"
train_set="voxceleb12_devs_sp_voxlingua107"
valid_set="voxceleb1_test"
test_sets="voxceleb1_test"
feats_type="raw"
spk_stats_dir="exp/spk_lid_stats_16k"

./replrn.sh \
    --spk_config ${spk_config} \
    --feats_type ${feats_type} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    --spk_stats_dir ${spk_stats_dir} \
    --stage 5 \
    "$@"

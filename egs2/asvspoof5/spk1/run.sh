#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_bin_sampler_sasv_SKA_mel.yaml

train_set="voxceleb2_asvspoof5"
valid_set="dev"
cohort_set="dev"
test_sets="dev"
skip_train=false

feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    --skip_train ${skip_train} \
    "$@"

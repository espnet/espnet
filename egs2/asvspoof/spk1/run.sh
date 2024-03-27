#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_rawnet3.yaml

train_set="train"
valid_set="dev"
cohort_set="dev"
test_sets="test"
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

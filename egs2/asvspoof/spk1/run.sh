#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_RawNet3.yaml

train_set="train"
valid_set="valid"
cohort_set="test"
test_sets="test"
skip_train=true

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

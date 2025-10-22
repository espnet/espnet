#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_resnet34.yaml

train_set="cnceleb_train"
valid_set="cnceleb1_valid"
cohort_set="cnceleb_train"
test_sets="cnceleb1_eval"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    "$@"

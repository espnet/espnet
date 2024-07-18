#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_RawNet3.yaml

train_set="rats_train"
valid_set="rats_test"
test_sets="rats_test"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    --speed_perturb_factors "" \
    --apply_noise_rir_augment false \
    --skip_train true \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=validation
test_sets="validation "  # modify this after the test set is released

# Note: It is suggested that you skip stages 3 and 4 after finishing stage 1. Instead, just run
#   mkdir -p dump; cp -r data dump/raw
# This is to keep the diverse sampling rates obtained from stage 1.
# Then start stage 5 to collect stats for training preparation.
# Finally, start training in stage 6.

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_bsrnn_large_noncausal.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

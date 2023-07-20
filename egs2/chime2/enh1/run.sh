#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# NOTE1: The reference signals in spk1.scp in train and devel subsets are reverberated.
# NOTE2: No reference signals are available for test subsets.

train_set=train
valid_set=devel
test_sets="test"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k



train_set=train
valid_set=valid
test_sets=testset

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --nj 16 \
    --inference_model "valid.loss.best.pth" \
    "$@"

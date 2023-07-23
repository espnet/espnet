#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./enh.sh \
    --nj 2 \
    --inference_nj 2 \
	--is_tse_task true \
    --fs 16k \
    --lang en \
    --enh_config ./conf/train_debug.yaml \
    --ref_num 1 \
    --train_set train_nodev \
    --valid_set test \
    --test_sets "train_dev test" \
    --inference_model "valid.loss.best.pth" \
    "$@"

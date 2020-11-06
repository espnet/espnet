#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k


train_set="train"
valid_set="dev"
test_sets="test "

./enh.sh \
<<<<<<< HEAD
    ----use_noise_ref true \
=======
>>>>>>> add librimix recipe
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang en \
    --ngpu 4 \
<<<<<<< HEAD
    --enh_config ./conf/tuning/train_enh_PSM.yaml \
=======
    --enh_config ./conf/tuning/train_enh_PSM_debug.yaml \
>>>>>>> add librimix recipe
    "$@"

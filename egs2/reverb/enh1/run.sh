#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# number of training channels
nch_train=2

train_set="tr_simu_${nch_train}ch"
valid_set="dt_multi_${nch_train}ch"
test_sets="dt_simu_8ch dt_real_8ch et_simu_8ch et_real_8ch"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs 16k \
    --spk_num 1 \
    --lang en \
    --ngpu 1 \
    --enh_config ./conf/tuning/train_enh_PSM.yaml \
    "$@"

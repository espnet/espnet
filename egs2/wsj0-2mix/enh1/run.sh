#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr
dev_set=cv
eval_sets="tt "

./enh.sh \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --ngpu 1 \
    --enh_config ./conf/tuning/train_enh_dorpout0.5.yaml \
    "$@"




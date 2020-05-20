#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --train_set train_nodev \
    --lm_config conf/train_lm.yaml \
    --dev_set train_dev \
    --eval_sets "test " \
    --srctexts "data/train_nodev/text" "$@"

#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr_hybrid.sh \
    --asr_config conf/train_asr_rnn.yaml \
    --mono_num_iters 1 \
    --skip_tri1 true \
    --skip_tri2 true \
    --tri3_num_iters 1 \
    --train_set train \
    --num_devsets 1 \
    --eval_sets "test" "$@"


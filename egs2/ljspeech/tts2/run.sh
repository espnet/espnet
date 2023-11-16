#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_no_dev
train_dev=dev
eval_set=eval1


./tts2.sh \
    --nj 4 \
    --inference_nj 4 \
    --lang en \
    --train_config conf/train_tacotron2_debug.yaml \
    --train_set ${train_set} \
    --valid_set ${train_dev} \
    --test_sets ${eval_set} \
    --srctexts "data/tr_no_dev/text" "$@"

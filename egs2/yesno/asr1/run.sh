#!/bin/bash

set -e
set -u
set -o pipefail

train_set=train_nodev
train_dev=train_dev
eval_set=test_yesno

asr_config=conf/train.yaml
decode_config=conf/decode.yaml

./asr.sh                                \
    --use_lm false                      \
    --token_type char                   \
    --feats_type raw                    \
    --asr_config "${asr_config}"        \
    --decode_config "${decode_config}"  \
    --train_set "${train_set}"          \
    --dev_set "${train_dev}"            \
    --eval_sets "${eval_set} "          \
    --srctexts "data/${train_set}/text" "$@"

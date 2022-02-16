#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="train_dev test_yesno"

asr_config=conf/train_asr.yaml
inference_config=conf/decode.yaml

./asr.sh                                        \
    --lang en                                   \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type char                           \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --lm_train_text "data/${train_set}/text" "$@"

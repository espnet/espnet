#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="train_dev test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode.yaml

lm_config=conf/train_lm_char.yaml
use_lm=true
use_wordlm=false
word_vocab_size=7184

./asr.sh                                        \
    --lang vi                                   \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type char                           \
    --use_lm ${use_lm}                          \
    --use_word_lm ${use_wordlm}                 \
    --word_vocab_size ${word_vocab_size}        \
    --lm_config "${lm_config}"                  \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --lm_train_text "data/${train_set}/text" "$@"

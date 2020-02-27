#!/bin/bash

set -e
set -u
set -o pipefail

train_set="train_nodev"
train_dev="train_dev"
eval_set="test"

asr_config=conf/train_asr.yaml
decode_config=conf/decode.yaml

lm_config=conf/train_lm_char.yaml
use_lm=true
use_wordlm=false
word_vocab_size=7184

./asr.sh                                        \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type char                           \
    --use_lm ${use_lm}                          \
    --use_word_lm ${use_wordlm}                 \
    --word_vocab_size ${word_vocab_size}        \
    --lm_config "${lm_config}"                  \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --dev_set "${train_dev}"                    \
    --eval_sets "${eval_set}"                   \
    --srctexts "data/${train_set}/text" "$@"

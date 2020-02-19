#!/bin/bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=11
nj=16
decode_asr_model=eval.loss.best.pth

train_set="train_nodev"
train_dev="train_dev"
eval_set="test"

asr_config=conf/train.yaml
decode_config=conf/decode.yaml

lm_config=conf/lm_char.yaml
use_lm=true
use_wordlm=false
word_vocab_size=7184

./asr.sh                                        \
    --stage ${stage}                            \
    --stop_stage ${stop_stage}                  \
    --nj ${nj}                                  \
    --audio_format wav                          \
    --decode_asr_model ${decode_asr_model}      \
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

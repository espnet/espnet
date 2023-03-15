#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=11
train_set=
valid_set=
test_sets=
asr_config=
inference_config=
lm_config=
use_word_lm=
word_vocab_size=

. ../utils/parse_options.sh

dir=$PWD
cd ../../asr1

./asr.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --lang en \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --feats_type raw \
    --feats_normalize uttmvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"

cd $dir

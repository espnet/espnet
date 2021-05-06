#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_reduced"
valid_set="dev5"
test_sets="dev5 test_set_iwslt2019"

asr_config=conf/train_asr_rnn.yaml
inference_config=conf/decode.yaml

feats_type=extracted

token_type=bpe

nlsyms=data/nlsyms

nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false

./asr.sh                                        \
    --lang en                                   \
    --feats_type ${feats_type}                  \
    --token_type ${token_type}                  \
    --nbpe ${nbpe}                              \
    --nlsyms_txt ${nlsyms}                      \
    --bpe_nlsyms ${bpe_nlsyms}                  \
    --use_lm ${use_lm}                          \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"

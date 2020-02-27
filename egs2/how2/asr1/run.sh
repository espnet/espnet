#!/bin/bash

set -e
set -u
set -o pipefail

train_set="train_reduced"
train_dev="dev5"
eval_set="test_set_iwslt2019"

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode.yaml

feats_type=extracted

token_type=bpe

nlsyms=data/nlsyms

nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false

./asr.sh                                        \
    --feats_type ${feats_type}                  \
    --token_type ${token_type}                  \
    --nbpe ${nbpe}                              \
    --nlsyms_txt ${nlsyms}                      \
    --bpe_nlsyms ${bpe_nlsyms}                  \
    --use_lm ${use_lm}                          \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --dev_set "${train_dev}"                    \
    --train_set "${train_set}"                  \
    --eval_sets "${eval_set}"                   \
    --srctexts "data/${train_set}/text" "$@"

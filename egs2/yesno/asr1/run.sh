#!/bin/bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=11
nj=16
decode_asr_model=eval.acc.best.pth 

train_set="train_nodev"
train_dev="train_dev"
eval_set="test_yesno"

asr_config=conf/train.yaml
decode_config=conf/decode.yaml

nlsyms_txt=data/nlsyms.txt

./asr.sh                                        \
    --stage ${stage}                            \
    --stop_stage ${stop_stage}                  \
    --nj ${nj}                                  \
    --decode_asr_model ${decode_asr_model}      \
    --feats_type fbank_pitch                    \
    --token_type char                           \
    --nlsyms_txt ${nlsyms_txt}                  \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --dev_set "${train_dev}"                    \
    --eval_sets "${eval_set}"                   \
    --srctexts "data/${train_set}/text" "$@"

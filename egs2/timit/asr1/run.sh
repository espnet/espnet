#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=11
ngpu=1
nj=16

train_set=train
dev_set=dev
eval_sets="test "

asr_config=conf/train.yaml
decode_config=conf/decode.yaml

trans_type=char  # char or phn

./asr.sh                                        \
    --stage ${stage}                            \
    --stop_stage ${stop_stage}                  \
    --ngpu ${ngpu}                              \
    --nj ${nj}                                  \
    --feats_type fbank_pitch                    \
    --token_type char                           \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --dev_set "${dev_set}"                      \
    --eval_sets "${eval_sets}"                  \
    --srctexts "data/${train_set}/text" "$@"    \
    --local_data_opts ${trans_type}

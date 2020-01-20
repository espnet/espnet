#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup_sp
dev_set=train_dev
eval_set="eval1 eval2 eval3"

asr_config=conf/tuning/train_rnn.yaml
decode_config=conf/tuning/decode_rnn.yaml
lm_config=conf/lm.yaml

./asr.sh \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_set}" \
    --srctexts "data/train_nodev/text" "$@"

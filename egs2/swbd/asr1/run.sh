#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
dev_set=train_dev
eval_sets="train_dev eval2000 rt03"
srctexts=data/${train_set}/text_lm

asr_config=conf/tuning/train_rnn.yaml
decode_config=conf/tuning/decode_rnn.yaml
lm_config=conf/lm.yaml

./asr.sh \
    --token_type bpe \
    --nbpe 2000 \
    --bpemode bpe \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "${srctexts}" "$@"

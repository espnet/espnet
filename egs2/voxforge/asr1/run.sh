#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=it # de, en, es, fr, it, nl, pt, ru
train_set="tr_${lang}"
dev_set="dt_${lang}"
eval_sets="et_${lang}"

lm_config=conf/lm_rnn/train.yaml
asr_config=conf/asr_rnn/train.yaml
decode_config=conf/asr_rnn/decode.yaml


./asr.sh \
    --local_data_opts "--lang ${lang}" \
    --token_type char \
    --feats_type fbank_pitch \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text" "$@"

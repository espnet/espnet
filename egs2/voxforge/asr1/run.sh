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


./asr.sh \
    --local_data_opts "--lang ${lang}" \
    --token_type char \
    --lm_config conf/lm_train.yaml \
    --asr_config conf/asr_train.yaml \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text" "$@"

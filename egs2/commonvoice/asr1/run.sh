#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


lang=cy # en de fr cy tt kab ca zh-TW it fa eu es ru

train_set=valid_train_${lang}
valid_set=valid_dev_${lang}
test_sets="valid_dev_${lang} valid_test_${lang}"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
decode_config=conf/decode_asr.yaml


./asr.sh \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe 150 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" "$@"

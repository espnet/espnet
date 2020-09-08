#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
valid_set=test_dev93
test_sets="test_dev93 test_eval92"

./asr.sh \
    --lang "en" \
    --nbpe 5000 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/train_lm.yaml \
    --asr_config conf/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
valid_set=test_dev93
test_sets="test_dev93 test_eval92"

./asr.sh \
    --lang en \
    --use_lm true \
    --token_type char \
    --nbpe 80 \
    --nlsyms_txt data/nlsyms.txt \
    --lm_config conf/train_lm_transformer.yaml \
    --asr_config conf/train_asr_conformer.yaml \
    --inference_config conf/decode.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train_si284/text" \
    --lm_train_text "data/train_si284/text data/local/other_text/text" "$@"

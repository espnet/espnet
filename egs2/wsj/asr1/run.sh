#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
dev_set=test_dev93
eval_sets="test_eval92 "

./asr.sh \
    --nbpe 5000 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/train_lm.yaml \
    --asr_config conf/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"

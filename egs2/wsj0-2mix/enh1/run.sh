#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr
dev_set=cv
eval_sets="tt "

./enh.sh \
    --nbpe 5000 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/train_lm.yaml \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --ngpu 2 \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"


    # --enh_config conf/train_asr_transformer.yaml \


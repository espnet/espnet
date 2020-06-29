#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
dev_set="dev"
eval_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_asr_transformer.yaml
lm_config=conf/train_lm.yaml
decode_config=conf/decode_asr.yaml

./asr.sh \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text data/local/other_text/text" "$@"

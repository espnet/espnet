#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_nodev"
dev_set="train_dev"
eval_sets="test_clean"

./asr.sh \
    --feats_type fbank_pitch \
    --token_type bpe \
    --nbpe 5000 \
    --use_lm false \
    --lm_config conf/train_lm.yaml \
    --asr_config conf/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/train_data_01/text" "$@"

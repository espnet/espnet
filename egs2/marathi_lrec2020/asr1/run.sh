#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="marathi_train"
valid_set="marathi_dev"
test_sets="marathi_dev marathi_test"

asr_config=conf/train_asr_transformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr_lm.yaml

./asr.sh \
    --ngpu 1 \
    --gpu_inference true \
    --feats_type raw \
    --token_type char \
    --use_lm true \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model latest.pth \
    --inference_lm valid.loss.best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --lm_train_text "dump/raw/${train_set}_sp/text" "$@"

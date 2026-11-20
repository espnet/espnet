#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="marathi_train"
valid_set="marathi_dev"
test_sets="marathi_dev marathi_test"

asr_config=conf/train.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode.yaml
inference_no_lm_config=conf/tuning/decode_no_lm.yaml

# Data prep, LM and ASR training, then decoding with the LM.
./asr.sh \
    --ngpu 1 \
    --gpu_inference true \
    --feats_type raw \
    --token_type char \
    --max_wav_duration 12 \
    --use_lm true \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --lm_train_text "data/${train_set}/text" "$@"

# Decoding of the same ASR model without the LM.
./asr.sh \
    --skip_data_prep true \
    --skip_train true \
    --ngpu 1 \
    --gpu_inference true \
    --feats_type raw \
    --token_type char \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_no_lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" "$@"

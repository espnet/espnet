#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/tuning/train_asr_transformer_multispkr.yaml
lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/decode.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --token_type "char" \
    --asr_task "asr" \
    --num_ref 2 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --inference_args "--multi_asr true" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2 data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

context=3
train_set="train_context${context}"
valid_set="valid_context${context}"
test_sets="test_context${context} valid_context${context}"

asr_config="conf/train_asr.yaml"
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type word \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.loss.ave.pth \
    --local_data_opts "--context ${context}" \
    --asr_stats_dir "exp/asr_stats_context${context}_raw_en_word_sp" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --feats-normalize null "$@"

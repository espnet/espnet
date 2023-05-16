#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"

asr_config=conf/train.yaml
inference_config=conf/decode.yaml

#if stage 10 fails, try setting the '--nj 1'
./asr.sh \
    --stage 1 \
    --stop_stage 13 \
    --nj 32 \
    --ngpu 1 \
    --gpu_inference true \
    --feats_type raw \
    --audio_format "flac" \
    --token_type bpe \
    --nbpe 2000 \
    --use_lm false \
    --max_wav_duration 30.0 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/local/text" \
    --lm_train_text "data/local/text"

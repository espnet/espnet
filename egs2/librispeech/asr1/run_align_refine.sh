#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/align_refine/train_conformer_align_refine.yaml
inference_config=conf/tuning/align_refine/decode.yaml
# inference_config=conf/tuning/align_refine/decode_ctc_only.yaml

./asr.sh \
    --stage 11 \
    --use_align_refine true \
    --lang en \
    --nj 64 \
    --ngpu 4 \
    --inference_nj 4 \
    --gpu_inference true \
    --use_lm false \
    --token_type bpe \
    --nbpe 400 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.loss.ave.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"

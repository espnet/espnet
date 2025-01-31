#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

nbpe=2000

min_wav_duration=0.3

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe "${nbpe}" \
    --max_wav_duration 30 \
    --speed_perturb_factors "1.0" \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config conf/train_lm.yaml \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

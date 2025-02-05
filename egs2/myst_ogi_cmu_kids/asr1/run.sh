#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="data_cmu/dev data_cmu/test data_ogi_spon/dev data_ogi_spon/test data_ogi_scripted/dev data_ogi_scripted/test data_myst/dev data_myst/test data_jibo"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

nbpe=5000

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe "${nbpe}" \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type "raw" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.cer_ctc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test_doreco"

encoder=transformer
frontend=xeus
if [[ -n "$frontend" ]]; then
    asr_config=conf/tuning/train_asr_${frontend}_${encoder}.yaml
    feats_type=extracted
else
    asr_config=conf/tuning/train_asr_${encoder}.yaml
    feats_type=raw
fi
inference_config=conf/decode_asr.yaml

./asr.sh \
    --token_type word \
    --max_wav_duration 30 \
    --use_lm false \
    --token_type word \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type "${feats_type}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"

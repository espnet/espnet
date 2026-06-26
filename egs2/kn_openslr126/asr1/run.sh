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
lm_config=conf/train_lm.yaml

token_type=bpe
nbpe=1000

./asr.sh \
    --lang kn \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --max_wav_duration 30 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

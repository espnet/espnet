#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_1h test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_wavlm_21_lr1e-4_warmup30k.yaml
inference_config=conf/decode_ctc0.3_beamsize5.yaml

./asr.sh \
    --ngpu 1 \
    --nbpe 6500 \
    --min_wav_duration 0.1 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --bpe_train_text "data/${train_set}/text" "$@"

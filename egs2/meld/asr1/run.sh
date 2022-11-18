#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="valid test"

asr_config=conf/train_asr.yaml
inference_config=conf/decoder_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --inference_config "${inference_config}" \
    --nbpe 850 \
    --token_type bpe\
    --bpe_nlsyms sadness,surprise,neutral,joy,anger,fear,disgust\
    --max_wav_duration 20 \
    --feats_normalize utterance_mvn\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

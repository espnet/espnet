#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="urdu"

train_set="$lang"/"train"
valid_set="$lang"/"valid"
test_sets="$lang/test $lang/test_known $lang/test_noisy $lang/test_known_noisy"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang $lang \
    --ngpu 1 \
    --stage 1 \
    --nj 32 \
    --inference_nj 32 \
    --token_type "bpe" \
    --nbpe 500 \
    --max_wav_duration 30 \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

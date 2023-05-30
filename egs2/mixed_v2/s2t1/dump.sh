#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=WenetSpeech/L
valid_set=WenetSpeech/DEV
test_sets="${valid_set}"

./s2t.sh \
    --stage 3 \
    --stop_stage 4 \
    --nj 128 \
    --feats_type raw \
    --audio_format flac.ark \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

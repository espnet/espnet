#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./uni_versa.sh \
    --nj 8 \
    --inference_nj 2 \
    --dumpdir dump_ark \
    --audio_format flac.ark \
    --train_config conf/train_universa.yaml \
    --inference_config conf/decode_universa.yaml \
    --dumpdir dump_ark \
    --train_set train \
    --valid_set dev \
    --test_sets "dev test" \
    --nbpe 500  \
    --ngpu 1 "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./uni_versa.sh \
    --audio_format flac.ark \
    --nj 10 \
    --inference_nj 10 \
    --train_config conf/train_universa.yaml \
    --inference_config conf/decode_universa.yaml \
    --dumpdir dump_ark \
    --train_set train_temp \
    --valid_set dev_temp \
    --test_sets "dev_temp test_temp" \
    --nbpe 500  \
    --ngpu 1 "$@"

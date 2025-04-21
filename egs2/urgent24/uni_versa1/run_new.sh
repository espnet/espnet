#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./uni_versa.sh \
    --nj 8 \
    --inference_nj 8 \
    --train_config conf/train_universa.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set train_utmos \
    --valid_set dev_temp \
    --test_sets "dev_temp test_temp" \
    --tag utmos_only \
    --dumpdir dump_ark \
    --audio_format flac.ark \
    --nbpe 500  \
    --inference_nj 1 \
    --gpu_inference true \
    --ngpu 1 "$@"

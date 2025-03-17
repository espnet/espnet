#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_set="train_960"
valid_set="dev"

train_config=conf/tuning/train_hubert.yaml

./ssl.sh \
    --ngpu 4 \
    --token_type word \
    --token_list ../hubert1/data/en_token_list_kmeans_iter1_espnet_hubert_500clusters/word/tokens.txt \
    --lang "en" \
    --nj 32 \
    --max_wav_duration 30 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

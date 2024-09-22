#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=""
test_sets=""

bpe_opts="--subword_choice huggingface --subword_model google/gemma-2b-it"

# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stop_stage 5 \
    --task "textlm" \
    --data_name dolma \
    --fs 16000 \
    --ngpu 8 \
    --nj 32 \
    --train_config conf/train_foo.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${bpe_opts} \
    "$@"

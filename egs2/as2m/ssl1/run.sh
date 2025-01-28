#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=0

train_set="train"
valid_set="eval"

timestamp=$(date "+%Y%m%d.%H%M%S")

timestamp=20250123.112306
ssl_tag=t1.${timestamp}

train_config=conf/pretrain_beats_as2m.yaml
storage_dir=/compute/babel-11-13/sbharad2/beats_pretraining

mkdir -p "${storage_dir}"

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag ${ssl_tag} \
    --n_targets 1024 \
    --expdir "${storage_dir}/exp" \
    --dumpdir "${storage_dir}/dump" \
    --datadir "${storage_dir}/data" \
    --stage 6 \
    --stop_stage 6 \
    --ngpu 8 \
    --num_nodes 1 \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 11 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

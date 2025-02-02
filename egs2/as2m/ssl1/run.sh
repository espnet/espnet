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
ssl_tag=iter0${timestamp}

train_config=conf/pretrain_beats_as2m.yaml
storage_dir=.
mkdir -p "${storage_dir}"

# 1-3 : cpu
# 4: gpu

# 5: cpu
# 6: gpu

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag ${ssl_tag} \
    --n_targets 1024 \
    --expdir "${storage_dir}/exp" \
    --dumpdir "${storage_dir}/dump" \
    --datadir "${storage_dir}/data" \
    --stage 5 \
    --stop_stage 6 \
    --ngpu 1 \
    --num_nodes 1 \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 11 \
    --tokenizer_inference_batch_size 128 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=1  # 1 iterations is enough for base model

n_clusters_iter0=1024

# Extract mfcc feature for k-means clustering to generate pseudo targets
feature_iter0="mfcc"
layer_iter0="0"

train_set="train_as2m"
valid_set="eval_as2m"

train_config_iter0=conf/train_ssl_beats_base_as2m_it0.yaml
storage_dir=./
local_data_opts=

./beats.sh \
    --portion_km -1 \
    --gpu_dump_feature true \
    --alignment_phoneme_dir "./data/librispeech_phoneme_alignment" \
    --speech_fold_length 1600 \
    --text_fold_length 600 \
    --expdir "${storage_dir}/exp" \
    --dumpdir "${storage_dir}/dump" \
    --datadir "${storage_dir}/data" \
    --local_data_opts ${local_data_opts} \
    --ngpu 8 \
    --num_nodes 1 \
    --lang "en" \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 10 \
    --train_configs "${train_config_iter0}" \
    --n_clusters "${n_clusters_iter0}" \
    --features_km "${feature_iter0}" \
    --layers_km "${layer_iter0}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

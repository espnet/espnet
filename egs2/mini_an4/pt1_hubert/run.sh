#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh


pretrain_start_iter=0
pretrain_stop_iter=0

n_clusters_iter0=100

feature_iter0="mfcc"

train_set="train_nodev"
valid_set="train_dev"

pretrain_config_iter0=conf/train_asr_hubert_base_pretrain_it0.yaml

./pt1_hubert.sh \
    --lang en \
    --pretrain_start_iter "${pretrain_start_iter}"\
    --pretrain_stop_iter "${pretrain_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30 \
    --pretrain_configs "${pretrain_config_iter0}" \
    --n_clusters "${n_clusters_iter0}" \
    --features_km "${feature_iter0}" \
    --portion_km 1.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

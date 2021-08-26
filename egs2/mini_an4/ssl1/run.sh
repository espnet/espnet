#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


pretrain_start_iter=0
pretrain_stop_iter=1

n_clusters_iter0=10
n_clusters_iter1=10

feature_iter0="mfcc"
feature_iter1="hubert6"

train_set="train_nodev"
valid_set="train_dev"

pretrain_config_iter0=conf/train_asr_hubert_base_pretrain_it0.yaml
pretrain_config_iter1=conf/train_asr_hubert_base_pretrain_it1.yaml

./hubert.sh \
    --lang en \
    --pretrain_start_iter "${pretrain_start_iter}"\
    --pretrain_stop_iter "${pretrain_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30 \
    --pretrain_configs "${pretrain_config_iter0} ${pretrain_config_iter1}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1}" \
    --features_km "${feature_iter0} ${feature_iter1}" \
    --portion_km 1.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"

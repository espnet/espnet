#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=1  # 1 iteration is enough for the base model

n_clusters_iter0=100
n_clusters_iter1=500

# Iteration 0: MFCC features for the k-means that bootstraps the pseudo targets
feature_iter0="mfcc"
layer_iter0="0"
# Iteration 1: latent features from transformer layer 6 of the WavLM model
# pre-trained in iteration 0
feature_iter1="espnet_wavlm"
layer_iter1="6"

train_set="train_960"
valid_set="dev"

train_config_iter0=conf/tuning/train_ssl_torchaudiowavlm_base_960h_pretrain_it0.yaml
train_config_iter1=conf/tuning/train_ssl_torchaudiowavlm_base_960h_pretrain_it1.yaml

# torch.compile. Measured on WavLM large, 8 x H100, 10,000 steps:
# 4727 s -> 3531 s (-25.3%) after a ~5 min one-off compile. It does NOT increase
# power draw (494.9 vs 501.9 W per GPU) -- it finishes the same work sooner.
# The configs already set this; the variable is here so it can be turned off
# from the command line without editing them: ./run.sh --use_torch_compile false
use_torch_compile=true

./wavlm.sh \
    --ngpu 8 \
    --num_nodes 1 \
    --lang "en" \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30.01 \
    --train_configs "${train_config_iter0} ${train_config_iter1}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1}" \
    --features_km "${feature_iter0} ${feature_iter1}" \
    --layers_km "${layer_iter0} ${layer_iter1}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --portion_km 0.1 \
    --gpu_dump_feature true \
    --alignment_phoneme_dir "./data/librispeech_phoneme_alignment" \
    --wavlm_args "--use_torch_compile ${use_torch_compile}" "$@"

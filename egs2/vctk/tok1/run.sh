#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

# ASR supervision and waveform preparation.
token_type=bpe
nbpe=5000
speed_perturb_factors="0.9 1.0 1.1"

# Differentiable tokenizer initialization.
use_default_centroid=true
kmeans_feature=wavlm_large/21
nclusters=2000
kmeans_portion=1.0

tok_task=gan_tok
tok_config=conf/train_gan_default.yaml

./tok.sh \
    --ngpu 4 \
    --nj 32 \
    --inference_nj 32 \
    --feats_type raw \
    --audio_format flac \
    --fs 16k \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_default_centroid "${use_default_centroid}" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --kmeans_portion "${kmeans_portion}" \
    --tok_task "${tok_task}" \
    --tok_config "${tok_config}" \
    "$@"

#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Datasets
train_set="train_minian4_nodev"
valid_set="train_minian4_dev"

test_sets="test_minian4"

tsne_set="test_minian4"

# Train
feats_type="raw"
config_dir="conf/default_ecapa_baseline.yaml"
exp_dir="exp"

# Inference
inference_model="valid.accuracy.ave.pth"
inference_batch_size=1


./lid.sh \
    --feats_type "${feats_type}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" \
    --inference_model "${inference_model}" \
    --inference_batch_size "${inference_batch_size}" \
    --extract_embd true \
    --nj 2 \
    --ngpu 0 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --expdir "${exp_dir}" \
    --lid_config "${config_dir}" \
    --perplexity 1 \
    --max_iter 250 \
    "$@"

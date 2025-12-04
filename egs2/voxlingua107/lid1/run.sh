#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Datasets
train_set="train_voxlingua107"
valid_set="dev_voxlingua107"

test_sets=""
id_test_set="dev_voxlingua107"
test_sets+="${id_test_set} "

tsne_set="dev_voxlingua107"

# Train
feats_type="raw"
config_dir="conf/mms_ecapa_baseline.yaml"
exp_dir="exp_voxlingua107_raw"

# Inference
inference_model="valid.accuracy.best.pth"
inference_batch_size=4
max_utt_per_lang_for_tsne=100


./lid.sh \
    --feats_type "${feats_type}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" \
    --inference_model ${inference_model} \
    --inference_batch_size ${inference_batch_size} \
    --extract_embd true \
    --checkpoint_interval 1000 \
    --expdir "${exp_dir}" \
    --lid_config "${config_dir}" \
    --max_utt_per_lang_for_tsne ${max_utt_per_lang_for_tsne} \
    "$@"

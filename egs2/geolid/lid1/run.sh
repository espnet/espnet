#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Datasets
train_set="train_voxlingua107_lang"
valid_set="dev_voxlingua107_lang"

# Additional test corpora can be selected with --test_sets after preparation.
test_sets="dev_voxlingua107_lang"

tsne_set="dev_voxlingua107_lang"

# Train
feats_type="raw"
config_dir="conf/train.yaml"
exp_dir="exp_voxlingua107_only"

# Inference
inference_model="valid.accuracy.best.pth"
inference_batch_size=1
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
    --nj 8 \
    --ngpu 1 \
    --expdir "${exp_dir}" \
    --lid_config "${config_dir}" \
    --max_utt_per_lang_for_tsne ${max_utt_per_lang_for_tsne} \
    "$@"

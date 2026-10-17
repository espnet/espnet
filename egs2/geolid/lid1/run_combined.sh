#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Datasets
train_set="train_all_no_filter_lang"
valid_set="dev_ml_superb2_lang"

test_sets=""

# In-domain test set(s)
id_test_set="
dev_voxlingua107_lang \
test_voxpopuli_lang \
test_fleurs_lang \
dev_ml_superb2_lang \
dev_dialect_ml_superb2_lang \
dev_babel_over_10s_lang
"
test_sets+="${id_test_set} "

tsne_set="train_all_no_filter_lang"

# Train
feats_type="raw"
config_dir="conf/combined/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml"
exp_dir="exp_combined"

# Inference
inference_model="valid.accuracy.best.pth"
inference_batch_size=4
max_utt_per_lang_for_tsne=100

./lid.sh \
    --feats_type ${feats_type} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" \
    --inference_model ${inference_model} \
    --inference_batch_size ${inference_batch_size} \
    --extract_embd false \
    --checkpoint_interval 1000 \
    --nj 8 \
    --ngpu 1 \
    --expdir "${exp_dir}" \
    --lid_config ${config_dir} \
    --max_utt_per_lang_for_tsne ${max_utt_per_lang_for_tsne} \
    "$@"

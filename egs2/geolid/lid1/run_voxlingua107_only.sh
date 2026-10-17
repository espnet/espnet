#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Datasets
train_set="train_voxlingua107_lang"
valid_set="dev_voxlingua107_lang"

test_sets=""

# In-domain test set(s)
id_test_set="dev_voxlingua107_lang"
test_sets+="${id_test_set} "

# Out-of-domain test set(s)
ood_test_set="
test_voxpopuli_lang \
test_fleurs_lang \
dev_ml_superb2_lang \
dev_dialect_ml_superb2_lang \
dev_babel_over_10s_lang
"
test_sets+="${ood_test_set} "

tsne_set="dev_voxlingua107_lang"

# Train
feats_type="raw"
# Four configs provided, each for a different conditioning projection layer type:
# 1. Shared + Trainable:
#    conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml
# 2. Shared + Frozen:
#    conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_frozen.yaml
# 3. Independent + Trainable:
#    conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_trainable.yaml
# 4. Independent + Frozen:
#    conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_frozen.yaml
config_dir="conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml"
exp_dir="exp_voxlingua107_only"

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
    --nj 8 \
    --ngpu 1 \
    --expdir "${exp_dir}" \
    --lid_config "${config_dir}" \
    --max_utt_per_lang_for_tsne ${max_utt_per_lang_for_tsne} \
    "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test"
cls_config=conf/beats_cls_vggsound.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=${timestamp}

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --gpu_inference false \
    --feats_normalize uttmvn \
    --use_lightning false \
    --stage 1 \
    --stop_stage 10 \
    --nj 10 \
    --label_fold_length 1 \
    --inference_nj 1 \
    --inference_model valid.acc.ave_1best.pth \
    --classification_type multi-class \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="val"
test_sets="test"
cls_config=conf/tuning/beats_cls_it3_e60_mixup02_emap.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=${timestamp}

# <=30s audio is 36743/36796 in train, 4165/4170 in val
max_wav_duration=30

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --use_lightning true \
    --gpu_inference true \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 10 \
    --nj 32 \
    --label_fold_length 200 \
    --max_wav_duration "${max_wav_duration}" \
    --inference_nj 16 \
    --inference_model valid.epoch_mAP.ave.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="eval"
test_sets="eval"
cls_config=conf/beats_cls.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"


./cls.sh \
    --cls_tag "${timestamp}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --gpu_inference true \
    --use_lightning true \
    --feats_normalize uttmvn \
    --ngpu 1 \
    --stage 1 \
    --stop_stage 10 \
    --nj 32 \
    --speech_fold_length 160000 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --max_wav_duration 12 \
    --inference_model valid.epoch_mAP.ave_1best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

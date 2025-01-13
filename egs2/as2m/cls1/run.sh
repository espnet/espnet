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
mynametag=balanced.unfrozen-beats_iter3p2m.${timestamp}
storage_dir=/compute/babel-11-13/sbharad2/beats_run/as2m_balanced_dynamic # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --feats_normalize uttmvn \
    --stage 5 \
    --stop_stage 10 \
    --ngpu 8 \
    --gpu_inference true \
    --nj 10 \
    --speech_fold_length 160000 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --inference_model valid.mAP.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

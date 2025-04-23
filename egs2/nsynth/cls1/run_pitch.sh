#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test"
cls_config=conf/beats_cls.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=${timestamp}

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"


./cls.sh \
    --cls_tag "${mynametag}" \
    --local_data_opts pitch \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --gpu_inference true \
    --feats_normalize uttmvn \
    --ngpu 1 \
    --stage 1 \
    --stop_stage 10 \
    --nj 10 \
    --speech_fold_length 480000 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

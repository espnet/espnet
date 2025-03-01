#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="humbugdb.train"
valid_set="humbugdb.dev"
test_sets="humbugdb.test"
cls_config=conf/beats_humbugdb.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=humbugdb.${timestamp}
storage_dir=.
mkdir -p "${storage_dir}"

./cls.sh \
    --local_data_opts "humbugdb" \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp/humbugdb" \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 10 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 10 \
    --speech_fold_length 160000 \
    --label_fold_length 5 \
    --max_wav_duration 32 \
    --inference_nj 1 \
    --inference_model valid.acc.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

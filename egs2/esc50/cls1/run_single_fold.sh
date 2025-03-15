#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

speech_fold_length=100000 # 6.25 sec, because audio is 5 sec each.
storage_dir=.
fold=1
cls_config=conf/beats_cls.yaml
mynametag=fast.fold

expdir=${storage_dir}/exp
dumpdir=${storage_dir}/dump
datadir=${storage_dir}/data

mkdir -p ${expdir}
mkdir -p ${dumpdir}
mkdir -p ${datadir}
train_set="train"
valid_set="val"
test_set="val"
./cls.sh \
    --local_data_opts "${fold}" \
    --cls_tag "${mynametag}${fold}" \
    --nj 8 \
    --datadir "${datadir}" \
    --dumpdir "${dumpdir}" \
    --expdir "${expdir}" \
    --ngpu 1 \
    --stage 1 \
    --gpu_inference true \
    --decoding_batch_size 128 \
    --speech_fold_length ${speech_fold_length} \
    --label_fold_length 1 \
    --feats_normalize utterance_mvn \
    --inference_nj 1 \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" "$@" &

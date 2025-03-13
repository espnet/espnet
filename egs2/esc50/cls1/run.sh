#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

speech_fold_length=1000 # 6.25 sec, because audio is 5 sec each.
storage_dir=.
n_folds=5 # This runs all 5 folds in parallel, take care.
cls_config=conf/beats_cls.yaml
mynametag=fast.fold

# NOTE(shikhar): Abusing variable lang to store fold number.
for fold in $(seq 1 $n_folds); do
    expdir=${storage_dir}/exp${fold}
    dumpdir=${storage_dir}/dump${fold}
    datadir=${storage_dir}/data${fold}

    mkdir -p ${expdir}
    mkdir -p ${dumpdir}
    mkdir -p ${datadir}
    train_set="train${fold}"
    valid_set="val${fold}"
    test_set="val${fold}"
    ./cls.sh \
        --local_data_opts "${fold}" \
        --cls_tag "${mynametag}${fold}" \
        --nj 8 \
        --datadir "${datadir}" \
        --dumpdir "${dumpdir}" \
        --expdir "${expdir}" \
        --ngpu 1 \
        --stage 1 \
        --speech_fold_length ${speech_fold_length} \
        --label_fold_length 1 \
        --feats_normalize utterance_mvn \
        --inference_nj 8 \
        --cls_config "${cls_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_set}" "$@" &
done
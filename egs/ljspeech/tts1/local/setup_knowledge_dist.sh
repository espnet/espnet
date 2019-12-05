#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Setup json files for knowledge distillation training in FastSpeech

# shellcheck disable=SC1091
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic setting
stage=0
stop_stage=100
backend=pytorch
nj=32
dumpdir=dump
verbose=1

# teacher model related
teacher_model_path=
decode_config=conf/decode_for_knowledge_dist.yaml

# data related
trans_type=phn
dict=
train_set=phn_train_no_dev
dev_set=phn_dev

# filtering related
do_filtering=false
focus_rate_thres=0.65

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check arguments
if [ -z "${teacher_model_path}" ]; then
    echo "you must set teacher_model_path." 2>&1
    exit 1;
fi
if [ -z "${dict}" ]; then
    echo "you must set dict." 2>&1
    exit 1;
fi

expdir=$(dirname "$(dirname "${teacher_model_path}")")
outdir=${expdir}/outputs_$(basename ${teacher_model_path})_$(basename ${decode_config%.*})
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Decoding for knowledge distillation"
    pids=() # initialize pids
    for name in ${dev_set} ${train_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${teacher_model_path} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/durations.$n.scp" || exit 1;
        done > ${outdir}/${name}/durations.scp
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/focus_rates.$n.scp" || exit 1;
        done > ${outdir}/${name}/focus_rates.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "successfully finished decoding."
fi

feat_tr_dir=${outdir}/dump/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${outdir}/dump/${dev_set}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Making json for knowledge distillation"
    # perform filtering
    if ${do_filtering}; then
        # TODO(kan-bayashi): implement filtering
        echo "Not yet supported."
        exit 1;
    fi

    # make data directory for knowledge distillation
    for name in ${dev_set} ${train_set}; do
        # make new data directory for knowledge distillation
        [ ! -e ${outdir}/data ] && mkdir -p ${outdir}/data
        utils/copy_data_dir.sh data/${name} ${outdir}/data/${name}
        cp ${outdir}/${name}/feats.scp ${outdir}/data/${name}/feats.scp
        utils/fix_data_dir.sh ${outdir}/data/${name}
    done

    # make new json for knowledge distillation
    data2json.sh --feat ${outdir}/data/${train_set}/feats.scp --trans_type ${trans_type} \
         ${outdir}/data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${outdir}/data/${dev_set}/feats.scp --trans_type ${trans_type} \
         ${outdir}/data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    local/update_json.sh ${feat_tr_dir}/data.json ${outdir}/${train_set}/durations.scp
    local/update_json.sh ${feat_dt_dir}/data.json ${outdir}/${dev_set}/durations.scp
    touch ${outdir}/.done
    echo "successfully finished making new json."
fi

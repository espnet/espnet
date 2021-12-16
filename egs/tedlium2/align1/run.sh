#!/usr/bin/env bash

# CTC segmentation example recipe

# Copyright 2017, 2020 Johns Hopkins University (Shinji Watanabe, Xuankai Chang)
# 2020, Technische Universität München, Authors: Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
python=python3
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
verbose=1      # verbose option

# feature configuration
do_delta=false
cmvn=

# Parameters for CTC alignment
# The subsampling factor depends on whether the encoder uses subsampling
subsampling_factor=4
# minium confidence score in log space - may need adjustment depending on data and model, e.g. -1.5 or -5.0
min_confidence_score=-5.0


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

align_set="dev test"
models=tedlium2.rnn.v2

download_dir=models
align_model=
align_config=
api=v1
dict=

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Model Download"
    local/download_data.sh
    . local/download_model.sh
fi


# Download trained models
if [ -z "${cmvn}" ]; then
    cmvn=$(find -L ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${align_model}" ]; then
    align_model=$(find -L ${download_dir}/${models} -name "model*.best*" | head -n 1)
fi
if [ -z "${align_config}" ]; then
    align_config=$(find -L ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi
if [ -z "${dict}" ]; then
    dict=$(find -L ${download_dir}/${models}/data/lang_*char -name "*.txt" | head -n 1) || dict=""

    if [ -z "${dict}" ]; then
        mkdir ${download_dir}/${models}/data/lang_autochar/ -p
        model_config=$(find -L ${download_dir}/${models}/exp/*/results/model.json | head -n 1)
        dict=${download_dir}/${models}/data/lang_autochar/dict.txt
        ${python} -c 'import json,sys;obj=json.load(sys.stdin);[print(char + " " + str(i + 1)) for i, char in enumerate(obj[2]["char_list"])]' > ${dict} < ${model_config}
    fi
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "CMVN file not found: ${cmvn}"
    exit 1
fi
if [ ! -f "${align_model}" ]; then
    echo "E2E model file not found: ${align_model}"
    exit 1
fi
if [ ! -f "${align_config}" ]; then
    echo "Config file not found: ${align_config}"
    exit 1
fi
if [ ! -f "${dict}" ]; then
    echo "Dictionary not found: ${dict}"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/prepare_data.sh
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${align_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done


    # dump features for training
    for rtask in ${align_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
            data/${rtask}/feats.scp ${cmvn} exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    for rtask in ${align_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp  \
            data/${rtask} ${dict}  > ${feat_recog_dir}/data.json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Alignments using CTC segmentation"

    for rtask in ${align_set}; do
	    # results are written to data/$rtask/aligned_segments
        ${python} -m espnet.bin.asr_align \
            --config ${align_config} \
            --ngpu ${ngpu} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --data-json ${dumpdir}/${rtask}/delta${do_delta}/data.json \
            --model ${align_model} \
            --subsampling-factor ${subsampling_factor} \
            --api ${api} \
            --utt-text data/${rtask}/utt_text \
            --output data/${rtask}/aligned_segments || exit 1;
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Removing utterances with low confidence scores"

    for rtask in ${align_set}; do
        unfiltered=data/${rtask}/aligned_segments
        filtered=data/${rtask}/aligned_segments_clean

        awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${unfiltered} > ${filtered}
        echo "Written $(wc -l < ${filtered}) of $(wc -l < ${unfiltered}) utterances to ${filtered}"
    done
fi

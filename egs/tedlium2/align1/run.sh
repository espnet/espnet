#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
python=python3
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
cmvn=

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_trim_sp
train_dev=dev_trim
recog_set="dev test"
models=tedlium2.rnn.v2


download_dir=models
align_model=
align_config=
align_dir=align
api=v1
dict=

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    echo "stage -1: Data Download"
    local/download_data.sh
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Model Download"
    . local/download_model.sh
fi


# Download trained models
if [ -z "${cmvn}" ]; then
    cmvn=$(find ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${align_model}" ]; then
    align_model=$(find ${download_dir}/${models} -name "model*.best*" | head -n 1)
fi
if [ -z "${align_config}" ]; then
    align_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi
if [ -z "${dict}" ]; then
    dict=$(find ${download_dir}/${models}/data/lang_*char -name "*.txt" | head -n 1) || dict=""

    if [ -z "${dict}" ]; then
        mkdir ${download_dir}/${models}/data/lang_autochar/ -p
        model_config=$(find ${download_dir}/${models}/exp/*/results/model.json | head -n 1)
        cat $model_config | python -c 'import json,sys;obj=json.load(sys.stdin);[print(char + " " + str(i + 1)) for i, char in enumerate(obj[2]["char_list"])]' > ${download_dir}/${models}/data/lang_autochar/dict.txt
        dict=${download_dir}/${models}/data/lang_autochar/dict.txt
    fi
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${align_model}" ]; then
    echo "No such E2E model: ${align_model}"
    exit 1
fi
if [ ! -f "${align_config}" ]; then
    echo "No such config file: ${align_config}"
    exit 1
fi
if [ ! -f "${dict}" ]; then
    echo "No such Dictionary file: ${dict}"
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
    for x in test dev; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done


    # dump features for training
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
            data/${rtask}/feats.scp ${cmvn} exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp  \
            data/${rtask} ${dict}  > ${feat_recog_dir}/data.json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Aligning"

    for rtask in ${recog_set}; do

        ${python} -m espnet.utils.ctc_align \
            --config ${align_config} \
            --ngpu ${ngpu} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --data-json ${dumpdir}/${rtask}/delta${do_delta}/data.json \
            --output data/${rtask}/aligned_segments \
            --model ${align_model} \
            --api ${api} \
            --utt-text data/dev/utt_text || exit 1;
    done
fi

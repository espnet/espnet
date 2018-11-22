#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


. ./path.sh
. ./cmd.sh

# Training options
backend=pytorch
stage=0
ngpu=0
debugmode=1
dumpdir=dump
N=0
verbose=0
resume=
seed=1
batchsize=15
maxlen_in=800
maxlen_out=150
epochs=15
tag=""

# Feature options
do_delta=false

# Encoder
etype=vggblstmp
elayers=4
eunits=768
eprojs=768
subsample=1_2_2_1_1

# Attention 
atype=location
adim=768
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# Decoder
dlayers=1
dunits=768

# Objective
mtlalpha=0.5
phoneme_objective_weight=0.0
# 2 or 1? This variable also affects the layer the adversarial objective plugs
# into
phoneme_objective_layer=""
lsm_type=unigram
lsm_weight=0.05
samp_prob=0.0

# Language prediction
predict_lang=""
predict_lang_alpha_scheduler="ganin"
langs_file=""

# Optimizer
opt=adadelta

# Decoding
beam_size=20
nbest=1
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
use_lm=false
decode_nj=32

# Training and adaptation languages
train_set="indonesian-notgt_train"
train_dev="indonesian-notgt_dev"
recog_set=""

adapt_langs=""

. ./utils/parse_options.sh || exit 1;

datasets=/export/b15/oadams/datasets-CMU_Wilderness

adapt_langs_train="${adapt_langs}_train"
adapt_langs_dev="${adapt_langs}_dev"

echo "train_set: ${train_set}"
echo "ngpu: ${ngpu}"

if [[ ${adapt_langs} ]]; then
    feat_tr_dir=${dumpdir}/${adapt_langs_train}_${train_set}/delta${do_delta}
    feat_dt_dir=${dumpdir}/${adapt_langs_dev}_${train_set}/delta${do_delta}
else
    feat_tr_dir=${dumpdir}/${train_set}_${train_set}/delta${do_delta}
    feat_dt_dir=${dumpdir}/${train_dev}_${train_set}/delta${do_delta}
fi

if [ ${stage} -le 1 ]; then
    echo "stage 1: Feature Generation"

    echo ${feat_tr_dir}
    echo ${feat_dev_dir}

    mkdir -p ${feat_tr_dir}
    mkdir -p ${feat_dt_dir}

    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    if [[ ${adapt_langs} ]]; then
        echo ${feat_tr_dir}
        echo ${feat_dev_dir}

        if [[ ! -e data/${train_set}/cmvn.ark ]]; then
            echo "Couldn't find train set CMVN feats at data/${train_set}/cmvn.ark."
            echo "Train a seed model first."
            echo "Exiting."
            exit
        fi

        for x in ${adapt_langs_train} ${adapt_langs_dev}; do
            if [[ ! -e ${dumpdir}/${x}_${train_set}/delta${do_delta}/feats.scp ]]; then
                steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50 --write_utt2num_frames true \
                    data/${x} exp/make_fbank/${x} ${fbankdir}
            fi
        done

        if [[ ! -e ${feat_tr_dir}/feats.scp ]]; then
            if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
            utils/create_split_dir.pl \
                /export/b{10,11,12,13}/${USER}/espnet-data/egs/cmu_wilderness/${exp_name}/dump/${adapt_langs_train}/delta${do_delta}/storage \
                ${feat_tr_dir}/storage
            fi
            dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
                data/${adapt_langs_train}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
        fi

        if [[ ! -e ${feat_dt_dir}/feats.scp ]]; then
            if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
            utils/create_split_dir.pl \
                /export/b{10,11,12,13}/${USER}/espnet-data/egs/cmu_wilderness/${exp_name}/dump/${adapt_langs_dev}/delta${do_delta}/storage \
                ${feat_dt_dir}/storage
            fi
            dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
                data/${adapt_langs_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
        fi

    else
        for x in ${train_set} ${train_dev}; do
            if [[ ! -e ${dumpdir}/${x}_${train_set}/delta${do_delta}/feats.scp ]]; then
                steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50 --write_utt2num_frames true \
                    data/${x} exp/make_fbank/${x} ${fbankdir}
            fi
        done

        if [[ ! -e ${feat_tr_dir}/feats.scp ]]; then
            # compute global CMVN
            compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

            exp_name=`basename $PWD`
            # dump features for training
            if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
            utils/create_split_dir.pl \
                /export/b{10,11,12,13}/${USER}/espnet-data/egs/cmu_wilderness/${exp_name}/dump/${train_set}/delta${do_delta}/storage \
                ${feat_tr_dir}/storage
            fi
            dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
                data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
        fi

        if [[ ! -e ${feat_dt_dir}/feats.scp ]]; then
            if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
            utils/create_split_dir.pl \
                /export/b{10,11,12,13}/${USER}/espnet-data/egs/cmu_wilderness/${exp_name}/dump/${train_dev}/delta${do_delta}/storage \
                ${feat_dt_dir}/storage
            fi
            dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
                data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
        fi

    fi

    for rtask in ${recog_set}; do
        if [[ ! -e data/${train_set}/cmvn.ark ]]; then
            echo "Couldn't find train set CMVN feats at data/${train_set}/cmvn.ark."
            echo "Train a seed model first."
            echo "Exiting."
            exit
        fi

        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}
        if [[ ! -e ${feat_recog_dir}/feats.scp ]]; then
            mkdir -p ${feat_recog_dir}
            dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
                data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
                ${feat_recog_dir}
        fi
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/${train_set}_non_lang_syms.txt

if [ ${stage} -le 2 ]; then
    echo "Stage 2: make json labels"
    # make json labels
    if [[ ${adapt_langs} ]]; then
        rtasks="${adapt_langs_train} ${adapt_langs_dev}"
    else
        rtasks="${train_set} ${train_dev} ${recog_set}"
    fi
    for rtask in ${rtasks} ${recog_set}; do
        echo $rtask
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}
        if [[ ! -e ${feat_recog_dir}/data.json ]]; then
            echo "Calling mkjson.py for $rtask"
            mkjson.py --non-lang-syms ${nlsyms} ${feat_recog_dir}/feats.scp \
                data/${rtask} ${dict} > ${feat_recog_dir}/data.json
        fi
    done
fi

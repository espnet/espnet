#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
# dumpdir=dump   # directory to dump full features
if [ $(hostname) = arcs22 ] || [ $(hostname) = amgp00 ] ; then
  dumpdir=/data/inaguma/espnet/egs/librispeech/asr1/dump
else
  dumpdir=/tmp/inaguma/espnet/egs/librispeech/asr1/dump
fi
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
# etype=blstmp     # encoder architecture type
# etype=vggblstmp     # encoder architecture type
etype=vggblstm     # encoder architecture type
# elayers=8
# elayers=5
elayers=3
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
# mtlalpha=0.5
mtlalpha=0

# minibatch related
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
# opt=adadelta
opt=adam
epochs=15

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
# ctc_weight=0.3
ctc_weight=0
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
# datadir=/export/a15/vpanayotov/data
datadir=/n/sd8/inaguma/corpus/librispeech/kaldi/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Set data size
datasize=100
# datasize=460
# datasize=960

gpu_ids=3

train_set=train_${datasize}
train_dev=dev_${datasize}
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part} || exit 1;
    done
fi

if [ ${stage} -le 0 ] && [ ! -e exp/.done_stage_0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g) || exit 1;
    done
    touch exp/.done_stage_0  # finish flag
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ! -e exp/.done_stage_1_${datasize}_$(hostname) ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    # for x in dev_clean test_clean dev_other test_other train_clean_100; do
    #     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/${x} exp/make_fbank/${x} ${fbankdir} || exit 1;
    # done

    if [ ${train_set} = train_100 ]; then
        cp -rf data/train_clean_100 data/train_100
    elif [ ${train_set} = train_360 ]; then
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/train_clean_360 exp/make_fbank/train_clean_360 ${fbankdir} || exit 1;
        cp -rf data/train_clean_360 data/train_360
    elif [ ${train_set} = train_460 ]; then
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/train_clean_360 exp/make_fbank/train_clean_360 ${fbankdir} || exit 1;
        utils/combine_data.sh data/train_460 data/train_clean_100 data/train_clean_360 || exit 1;
    elif [ ${train_set} = train_960 ]; then
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/train_clean_360 exp/make_fbank/train_clean_360 ${fbankdir} || exit 1;
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/train_other_500 exp/make_fbank/train_other_500 ${fbankdir} || exit 1;
        utils/combine_data.sh data/train_960 data/train_clean_100 data/train_clean_360 data/train_other_500 || exit 1;
    else
        echo "datasize is 100 or 460 or 960."
        exit 1;
    fi
    utils/combine_data.sh data/${train_dev} data/dev_clean data/dev_other || exit 1;

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 400 characters or no more than 0 characters
    # remove_longshortdata.sh --maxchars 400 data/train data/${train_set}
    # remove_longshortdata.sh --maxchars 400 data/dev data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark || exit 1;

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
            ${feat_tr_dir}/storage || exit 1;
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
            ${feat_dt_dir}/storage || exit 1;
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir} || exit 1;
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir} || exit 1;
    touch exp/.done_stage_1_${datasize}_$(hostname)  # finish flag
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ! -e exp/.done_stage_2_${datasize}_$(hostname) ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict} || exit 1;
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json || exit 1;
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json || exit 1;
    touch exp/.done_stage_2_${datasize}_$(hostname)  # finish flag
fi

# You can skip this and remove --rnnlm option in the recognition (stage 3)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]  && [ ! -e ${lmexpdir}/.done_training ]; then
    echo "stage 3: LM Preparation"
    dict=data/lang_1char/train_960_units.txt
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/train_960/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train.txt || exit 1;
    text2token.py -s 1 -n 1 data/dev/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt || exit 1;
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
        lmngpu=1
    else
        lmngpu=${ngpu}
    fi
    CUDA_VISIBLE_DEVICES=${gpu_id} ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${lmngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 40 \
        --batchsize 256 \
        --dict ${dict} || exit 1;
    touch {lmexpdir}/.done_training
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ! -e ${expdir}/.done_training ]; then
    echo "stage 4: Network Training"
    CUDA_VISIBLE_DEVICES=${gpu_ids} ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
   touch ${expdir}/.done_training
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/${train_set}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json.sh ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict} >> ${expdir}/RESULTS_${rtask}

    ) &
    done
    wait
    echo "Finished"
fi

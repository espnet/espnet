#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
ctctype=chainer
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=1.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# bpe
nbpe=500

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh 
. ./cmd.sh 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
train_dev=test_dev93
recog_set="test_dev93 test_eval92"

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_si284 test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/bpe/${train_set}_units_${nbpe}.txt
code=data/bpe/code_${nbpe}.txt
nlsyms=data/bpe/non_lang_syms.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/bpe
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text | learn_bpe.py -s ${nbpe} > ${code}
    cut -f 2- -d" " data/${train_set}/text | apply_bpe.py -c ${code} \
	    | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}
    
    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${code} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data_${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${code} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data_${nbpe}.json
fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs2048_bpe${nbpe}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    cut -f 2- -d" " data/${train_set}/text | apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans_${nbpe}.txt
    zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | grep -v "<" | tr [a-z] [A-Z] \
        | apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' >> ${lmdatadir}/train_others_${nbpe}.txt
    cat ${lmdatadir}/train_trans_${nbpe}.txt ${lmdatadir}/train_others_${nbpe}.txt | tr '\n' ' ' > ${lmdatadir}/train_${nbpe}.txt
    cut -f 2- -d" " data/${train_dev}/text | apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid_${nbpe}.txt
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --gpu ${gpu} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train_${nbpe}.txt \
        --valid-label ${lmdatadir}/valid_${nbpe}.txt \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_bpe${nbpe}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

# switch backend
if [[ ${backend} == chainer ]]; then
    train_script=asr_train.py
    decode_script=asr_recog.py
else
    train_script=asr_train_th.py
    decode_script=asr_recog_th.py
fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} ${expdir}/train.log \
        ${train_script} \
        --gpu ${gpu} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data_${nbpe}.json \
        --valid-label ${feat_dt_dir}/data_${nbpe}.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
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
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

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
        data2json.sh --bpecode ${code} ${data} ${dict} > ${data}/data_${nbpe}.json

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            ${decode_script} \
            --gpu ${gpu} \
            --recog-feat "$feats" \
            --recog-label ${data}/data_${nbpe}.json \
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

        score_sclite.sh --bpe true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
            
    ) &
    done
    wait
    echo "Finished"
fi


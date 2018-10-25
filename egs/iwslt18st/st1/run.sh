#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network archtecture
# encoder related
etype=vggblstm     # encoder architecture type
elayers=3
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=dot
adim=1024
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0

# minibatch related
batchsize=15
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=10
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/b08/inaguma/IWSLT


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

train_set=train_de
train_dev=dev2010_de
recog_set="dev2010_de tst2010_de tst2013_de tst2014_de tst2015_de"
eval_set=tst2018_de


if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in train dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
        local/download_and_untar.sh ${datadir} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/data_prep.sh ${datadir}/iwslt-corpus ${part}
    done

    # for IWSLT 2018 submission
    local/data_prep_eval.sh ${datadir}/IWSLT.tst2018
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x}_de exp/make_fbank/${x} ${fbankdir}
    done

    for x in train dev2010; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        utils_espnet/remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}_de data/${x}_de_tmp
        utils_espnet/remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}_en data/${x}_en_tmp

        # Match the number of utterances between EN and DE
        # extract commocn lines
        cut -f -1 -d " " data/${x}_de_tmp/segments > data/${x}_de_tmp/reclist1
        cut -f -1 -d " " data/${x}_en_tmp/segments > data/${x}_de_tmp/reclist2
        comm -12 data/${x}_de_tmp/reclist1 data/${x}_de_tmp/reclist2 > data/${x}_de_tmp/reclist

        utils_espnet/reduce_data_dir.sh data/${x}_de_tmp data/${x}_de_tmp/reclist data/${x}_de_trim
        utils_espnet/reduce_data_dir.sh data/${x}_en_tmp data/${x}_de_tmp/reclist data/${x}_en_trim
        utils/fix_data_dir.sh data/${x}_de_trim
        utils/fix_data_dir.sh data/${x}_en_trim
        rm -rf data/${x}_de_tmp
        rm -rf data/${x}_en_tmp
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}_trim/feats.scp data/${train_set}_trim/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/asr1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/asr1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    utils_espnet/dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}_trim/feats.scp data/${train_set}_trim/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    utils_espnet/dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}_trim/feats.scp data/${train_set}_trim/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        utils_espnet/dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}_trim/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/train_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    # The same dictinary between EN and DE
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/train_en/text data/train_de/text | utils_espnet/text2token.py -s 1 -n 1 | cut -f 2- -d " " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    utils_espnet/data2json.sh --feat ${feat_tr_dir}/feats.scp \
        data/${train_set}_trim ${dict} > ${feat_tr_dir}/data.json
    utils_espnet/data2json.sh --feat ${feat_dt_dir}/feats.scp \
        data/${train_dev}_trim ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        utils_espnet/data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 3)
lmexpdir=exp/train_rnnlm_${backend}_2layer_bs256_de
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_de
    mkdir -p ${lmdatadir}
    utils_espnet/text2token.py -s 1 -n 1 data/${train_set}_trim/text | cut -f 2- -d " " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train.txt
    utils_espnet/text2token.py -s 1 -n 1 data/${train_dev}_trim/text | cut -f 2- -d " " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 60 \
        --batchsize 256 \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
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
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --eps-decay 1 \
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        mkdir -p ${expdir}/${decode_dir}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        utils_espnet/splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        local/score_bleu.sh --word true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

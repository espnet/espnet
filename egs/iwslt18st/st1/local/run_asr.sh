#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=vggblstm     # encoder architecture type
elayers=5
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=location
adim=1024
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# regualrization option
samp_prob=0
lsm_type=unigram
lsm_weight=0.1
drop_enc=0.3
drop_dec=0.0
weight_decay=0

# minibatch related
batchsize=15
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=10
patience=3

lm_layers=2
lm_units=650
lm_opt=adam       # or sgd
lm_batchsize=256  # batch size in LM training
lm_epochs=60      # if the data size is large, we can reduce this
lm_maxlen=150     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=0.3
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# preprocessing related
case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
st_ted=/export/b08/inaguma/IWSLT
st_ted=~/corpus/iwslt18/data

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

train_set=train_nodevtest_sp.en
train_dev=train_dev.en
recog_set="dev.en test.en dev2010.en tst2010.en tst2013.en tst2014.en tst2015.en"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    if [ ! -d "data/train_nodevtest_sp.en" ]; then
        echo "run .run.sh first"
        exit 1
    fi

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/train_units_${case}.txt
nlsyms=data/lang_1char/non_lang_syms_${case}.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cat data/train_nodevtest_sp.*/text.${case} | grep sp1.0 | cut -f 2- -d " " | grep -o -P '&[^;]*;' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    # Share the same dictinary between source and target languages
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/train_nodevtest_sp.*/text.${case} | grep sp1.0 | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${case} --nlsyms ${nlsyms} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.${case}.json
    local/data2json.sh --feat ${feat_dt_dir}/feats.scp --text data/${train_dev}/text.${case} --nlsyms ${nlsyms} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data.${case}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        if [ ${rtask} = "dev.en" ] || [ ${rtask} = "test.en" ]; then
          local/data2json.sh --feat ${feat_recog_dir}/feats.scp --text data/${rtask}/text.${case} --nlsyms ${nlsyms} \
              data/${rtask} ${dict} > ${feat_recog_dir}/data.${case}.json
        else
          local/data2json.sh --feat ${feat_recog_dir}/feats.scp --no_text true \
              data/${rtask} ${dict} > ${feat_recog_dir}/data.${case}.json
        fi
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 3)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
fi
lmexpdir=exp/${train_set}_${case}_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_${train_set}
    mkdir -p ${lmdatadir}
    cat data/${train_set}/text.${case} | grep sp1.0 | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " \
        > ${lmdatadir}/train_${case}.txt
    cat data/${train_dev}/text.${case} | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " \
        > ${lmdatadir}/valid_${case}.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train_${case}.txt \
        --valid-label ${lmdatadir}/valid_${case}.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${case}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_lsm${lsm_weight}_drop${drop_enc}${drop_dec}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_wd${weight_decay}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.${case}.json \
        --valid-json ${feat_dt_dir}/data.${case}.json \
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
        --sampling-probability ${samp_prob} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --dropout-rate ${drop_enc} \
        --dropout-rate-decoder ${drop_dec} \
        --opt ${opt} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --patience ${patience} \
        --weight-decay ${weight_decay}
fi

exit 1

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        mkdir -p ${expdir}/${decode_dir}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.${case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight}

        if [ ${rtask} = "dev.en" ] || [ ${rtask} = "test.en" ]; then
          local/score_sclite.sh --case ${case} --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}
        else
          set=$(echo ${rtask} | cut -f -1 -d ".")
          local/score_sclite_reseg.sh --case ${case} --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict} ${st_ted} ${set}
        fi

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

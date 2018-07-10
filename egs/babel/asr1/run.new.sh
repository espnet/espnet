#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh


# Data settings 
do_delta=false # true when using CNN
do_cvn=true

dumpdir=dump   # directory to dump full features
dumpdir_name=delta${do_delta}cvn$do_cvn # 

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stage_last=100000 # end on the END
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot


# network archtecture
# encoder related
etype=blstmp # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300

# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# exp tag
tag="" # tag for managing experiments.

lang_id="101"

. utils/parse_options.sh || exit 1;

# Train test sets
train_set=train
train_dev=dev
recog_set="eval_${lang_id}"


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

if [ $stage -le 0 ] && [ $stage_last -ge 0 ]; then
  echo "stage 0: Setting up individual languages"
  # It will fill directories for all $lang_ids: data/${lang_id}/data 
  #     by train_${lang_id} dev_${lang_id} 
  # and combine them together into main data/{train,dev}
  ./local/setup_languages.sh --langs "${lang_id}" --recog "${lang_id}"
fi

fbankdir=fbank
if [ $stage -le 1 ] && [ $stage_last -ge 1 ]; then

    # Generate and dump features
    for x in ${train_set} ${train_dev} ${recog_set}; do
	local/makeanddump_fea.sh \
	    --data_in data/$x \
	    --data_fea ${fbankdir}/$x \
	    --data_dmp ${dumpdir}/$x/${dumpdir_name}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/${dumpdir_name}
feat_dt_dir=${dumpdir}/${train_dev}/${dumpdir_name}

#dict=data/lang_1char/${train_set}_units.txt
#nlsyms=data/lang_1char/non_lang_syms.txt
lang=data/lang

if [ ${stage} -le 2 ] && [ ${stage_last} -ge 2 ]; then
    echo "Creating dictionary: into  ${lang}"
    # --- Prepare $lang/train_unit.dct
    if [ ! -d $lang ]; then
	./local/make_dct.sh \
	    --lang $lang    \
	    --data "data/${train_set} data/${train_dev}"
    fi

    for x in ${train_set} ${train_dev} ${recog_set}; do
	./local/make_json.sh \
	    --lang $lang                    \
	    --data_in $fbankdir/$x --data ${dumpdir}/$x/${dumpdir_name}
    done
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

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict $lang/train_units.txt \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
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
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32
    
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}

        # split data
        #data=data/${rtask}
	data=${dumpdir}/${rtask}/$dumpdir_name

	# Copy data files for scoring and data split
	for f in utt2spk spk2utt text wav.scp segments; do
	    [ ! -e $data/${f} ] && cp data/$rtask/$f ${data}/
	done
	split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        #feats="ark,s,cs:apply-cmvn --norm-vars=true $fbankdir/${rtask}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        #if ${do_delta}; then
        #feats="$feats add-deltas ark:- ark:- |"
        #fi
        feats="ark:copy-feats scp:${sdata}/JOB/feats.scp ark:- |"

        # make json labels for recognition
	./local/make_json.sh \
            --lang $lang                    \
            --data_in $fbankdir/${rtask} --data $data

        #### use CPU for decoding
        ngpu=0

	# For rnnlm
	extra_opts=""
 
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --ctc-weight ${ctc_weight} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            ${extra_opts} &
        wait

	nlsyms=$lang/non_lang_syms.txt
	dict=$lang/train_units.txt
        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


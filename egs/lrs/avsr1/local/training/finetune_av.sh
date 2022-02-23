#!/bin/bash


# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
backend=pytorch			# set backend type
ngpu=1         			# number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            			# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      			# verbose option
nbpe=5000
bpemode=unigram
nj=32
do_delta=false
preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


# parameter handover
expdir=$1
dumptraindir=$2
dumpdecodedir=$3
PRETRAINEDMODEL=$4
lmexpdir=$5
noisetype=$6
dict=$7
bpemodel=$8
pretrainedav=$9


stage=0       # start from -1 if you need to start from data download
stop_stage=100

resume=	      # Resume the training from snapshot

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# define sets
train_set=Train
train_dev=Val
recog_evalset="-12 -9 -6 -3 0 3 6 9 12 clean reverb"

#### The features should already extracted and the language model should be already trained

if [ ! -d $expdir ]; then
    mkdir -p ${expdir}
fi

feat_tr_dir=$dumptraindir/${train_set}/delta${do_delta}
feat_dt_dir=$dumptraindir/${train_dev}/delta${do_delta}

# Tag name
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

# Replace files with custom files
rm -rf $MAIN_ROOT/espnet/finetuneav  || exit 1;
ln -rsf  local/training/finetuneav ${MAIN_ROOT}/espnet/finetuneav  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_avrms.py   || exit 1;
ln -rsf  local/training/finetuneav/asr_train_avrms.py $MAIN_ROOT/espnet/bin/asr_train_avrms.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_avrms.py  || exit 1;
ln -rsf  local/training/finetuneav/asr_recog_avrms.py $MAIN_ROOT/espnet/bin/asr_recog_avrms.py  || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Fintune Audio-Visual Network"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_avrms.py \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config $(change_yaml.py ${train_config} -o conf/finetuneav.yaml -a model-module=espnet.finetuneav.e2e_asr_transformer:E2E -a batch-size=1 -a epochs=10 -a transformer-lr=0.05 -a transformer-warmup-steps=2500)  \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname}av \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        --pretrain-video-extractor $PRETRAINEDMODEL  \
        --pretrain-av-model $pretrainedav/results/model.last10.avg.best 
fi

rm -rf ${expdir}/$noisetype
mkdir ${expdir}/$noisetype
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}  || exit 1;
    fi
    echo "stage 1: Decoding"
    for rtask in $recog_evalset; do   ####################### the file you want to decode
        pids=() # initialize pids
        (decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
         feat_recog_dir=${dumpdecodedir}/Test_${noisetype}/${rtask}/delta${do_delta}

         # split data
         splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json  || exit 1;

         #### use CPU for decoding
         ngpu=0
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_avrms.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose 1 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best  || exit 1;

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}  || exit 1;
	mv ${expdir}/${decode_dir} ${expdir}/$noisetype/${decode_dir}
         ) &
         pids+=($!) # store background pids

    
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    
        echo "Finished"
     done
fi
rm -rf $MAIN_ROOT/espnet/finetuneav  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_avrms.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_avrms.py  || exit 1;

exit 0


#!/usr/bin/env bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



# general configuration
backend=			# set backend type
ngpu=         			# number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=
N=            			# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=      			# verbose option
nbpe=
bpemode=
nj=
do_delta=
train_set=pretrain_Train
train_dev=Val
preprocess_config=
# audio model does not use ESPnet default code: 6 layer self-attention is added block before calculating ctc loss
train_config=  	      	 
lm_config=
decode_config=

[ -f path.sh ] && . ./path.sh # source the path.
. utils/parse_options.sh || exit 1;

# parameter handover
expdir=$1
dumptraindir=$2
dumpdecodedir=$3
lmexpdir=$4
noisetype=$5
dict=$6
bpemodel=$7


stage=0				# stage 0 for training and stage 1 for decoding
stop_stage=100			# stage at which to stop 
resume=        			# Resume the training from snapshot


# rnnlm related
lm_resume=        		# specify a snapshot file to resume LM training
lmtag=            		# tag for managing LMs

# decoding parameter
recog_model=model.acc.best 	# set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10



# exp tag
tag="" 				# tag for managing experiments.



# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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
rm -rf $MAIN_ROOT/espnet/trainaudio  || exit 1;
ln -rsf  local/training/trainaudio ${MAIN_ROOT}/espnet/trainaudio  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_audio.py   || exit 1;
ln -rsf  local/training/trainaudio/asr_train_audio.py $MAIN_ROOT/espnet/bin/asr_train_audio.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_audio.py  || exit 1;
ln -rsf  local/training/trainaudio/asr_recog_audio.py $MAIN_ROOT/espnet/bin/asr_recog_audio.py  || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Network training stage 0: Audio-only Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_audio.py \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config ${train_config} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname}_a \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

rm -rf ${expdir}/$noisetype
mkdir ${expdir}/$noisetype
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    echo "stage 1: Decoding"
    for rtask in $recog_evalset; do
        pids=() # initialize pids
        (decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=$dumpdecodedir/Test_${noisetype}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_audio.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --verbose 1 \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}  \
                --rnnlm ${lmexpdir}/rnnlm.model.best

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
         mv ${expdir}/${decode_dir} ${expdir}/$noisetype/${decode_dir}
        ) &
        pids+=($!) # store background pids

        
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        
        echo "Finished"
    done
fi
# remove custom files
rm -rf $MAIN_ROOT/espnet/trainaudio  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_audio.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_audio.py  || exit 1;

#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# general configuration
backend=pytorch			# set backend type
ngpu=1         			# number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            			# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      			# verbose option
nbpe=500
bpemode=unigram
nj=32
do_delta=false


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

# parameter handover
expdir=$1
dumpdecodedir=$2
lmexpdir=$3
dict=$4
bpemodel=$5


resume=        			# Resume the training from snapshot


preprocess_config=conf/specaug.yaml
train_config=conf/pretrainvideo/pretrainvideo.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        		# specify a snapshot file to resume LM training
lmtag=            		# tag for managing LMs

# decoding parameter
recog_model=model.last10.avg.best 	# set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10



# exp tag
tag="" # tag for managing experiments.



# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=pretrain
train_dev=Val
recog_evalset="-12 -9 -6 -3 0 3 6 9 12 clean reverb"

#### The features should already extracted and the language model should be already trained

if [ ! -d $expdir ]; then
    mkdir -p ${expdir}
fi

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
rm -rf $MAIN_ROOT/espnet/pretrainvideo  || exit 1;
ln -rsf  local/pretrainvideo ${MAIN_ROOT}/espnet/pretrainvideo  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_videopretrain.py   || exit 1;
ln -rsf  local/pretrainvideo/asr_train_videopretrain.py $MAIN_ROOT/espnet/bin/asr_train_videopretrain.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_videopretrain.py  || exit 1;
ln -rsf  local/pretrainvideo/asr_recog_videopretrain.py $MAIN_ROOT/espnet/bin/asr_recog_videopretrain.py  || exit 1;


echo "stage 1: Decoding"
for noisetype in noise saltandpepper music blur; do
rm -rf ${expdir}/$noisetype
mkdir ${expdir}/$noisetype 
    for rtask in $recog_evalset; do   ####################### the file you want to decode
        pids=() # initialize pids
        (decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
         feat_recog_dir=${dumpdecodedir}/Test_${noisetype}/${rtask}/delta${do_delta}

         # split data
         splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json  || exit 1;

         #### use CPU for decoding
         ngpu=0
         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
             asr_recog_videopretrain.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --verbose 1 \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/${recog_model}  \
                --rnnlm ${lmexpdir}/rnnlm.model.best  || exit 1;

         score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}  || exit 1;
         mv ${expdir}/${decode_dir} ${expdir}/$noisetype/${decode_dir}
         ) &
         pids+=($!) # store background pids

    
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    
        echo "Finished"
     done

done

rm -rf $MAIN_ROOT/espnet/pretrainvideo  || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_train_videopretrain.py   || exit 1;
rm -rf $MAIN_ROOT/espnet/bin/asr_recog_videopretrain.py  || exit 1;


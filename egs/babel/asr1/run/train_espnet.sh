#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh


# Data settings 
data_train=dump/train
data_dev=dump/dev
data_eval=

lang=data/lang_1char # for scoring

# general configuration
stage=2         # start from 0 if you need to start from data preparation
stage_last=100000 # end on the END
resume=        # Resume the training from snapshot

train_conf=conf/espnet.base.conf  # file where $train_opt and $tag variable is saved
eval_conf=conf/espnet.base.conf   # file where $eval_opt is defined

extra_train_opts=    # various opt which can be added on the and of training
extra_train_conf=/dev/null # various opts which can be saved in config file

extra_eval_opts=    # various opt which can be added on the and of training
extra_eval_conf=/dev/null # various opts which can be saved in config file


# exp tag
expdir=
tagflag=              # it will be add to the and of $tag # usefull for various input data

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train
source $train_conf 
[ -z $expdir ] && expdir=exp/$(basename ${data_train})_$tag${tagflag} # tag and training options are defined in $train_conf

mkdir -p ${expdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    echo " conf: $train_conf"
    echo " extra conf: $extra_train_conf"
    echo " {extra_train_opts}" 

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --outdir ${expdir}/results \
        --train-json ${data_train}/data.json \
        --valid-json ${data_dev}/data.json \
	${opt_train} \
	$(source $extra_train_conf) \
	${extra_train_opts}
fi

source $eval_conf 

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network testing"
    echo " conf: $eval_conf"
    echo " extra conf: $extra_eval_conf"
    echo " {extra_eval_opts}" 

if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=80

    for data in ${data_eval}; do
	(
            decode_dir=decode_$(basename ${data})_${tag_eval}
            # Check if needed data files are in target dir
            for f in utt2spk spk2utt text wav.scp segments; do
		[ ! -e ${data}/${f} ] && echo "Missing ${data}/${f}. Copy it." && exit 1
            done
	    [ ! -e ${data}/data.json ] && echo "Missing ${data}/data.json. Create it!!" && exit 1
            # split data

            splitjson.py --parts ${nj} ${data}/data.json

            #### use CPU for decoding
            ngpu=0
	    
            # For rnnlm
            extra_opts=""
	    
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
		asr_recog.py \
		--outdir ${expdir}/results \
		--model ${expdir}/results/model.${recog_model}  \
		--model-conf ${expdir}/results/model.conf  \
		--recog-json ${data}/split${nj}utt/data.JOB.json \
		--result-label ${expdir}/${decode_dir}/data.JOB.json \
		${opt_eval} \
		$(source $extra_train_conf) \
		${extra_eval_opts} &
        wait
	
        nlsyms=$lang/non_lang_syms.txt
        dict=$lang/train_units.txt
        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
echo "Finished"

#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a task of 10 language-indepent ASR used in
# S. Watanabe et al, "Language independent end-to-end architecture for
# joint language identification and speech recognition," Proc. ASRU'17, pp. 265--269 (2017)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0         # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
storage=/mnt/scratch01/tmp/karafiat/espnet/$RANDOM

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=5
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
#ctctype=chainer #warpctc
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
batchsize=50
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

# -- data
data_train=data/tr_102
data_dev=data/dt_102
data_eval=data/et_102

# dct
lang=data/lang_1char # dir where units and non lang symbols are expected

# -- MultNN info
multnn_dir=/mnt/matylda6/baskar/espnet_forked/egs/babel/asr1/exp/tr_babel10_blstmp_e5_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs50_mli800_mlo150
multnn_resume=$multnn_dir/results/snapshot_iter_12847

expname=karthick_v0__ft_102


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


dict=$lang/train_units.txt
nlsyms=$lang/non_lang_syms.txt


## Adaptation of the language independant (LI) trained model towards target languages
#           ***************** has to be implemented ******************

mexpdir=exp/$expname
m2expdir=${mexpdir}_it2

if [ ${stage} -le 1 ]; then
    echo "stage 1: NN language transfer"

    epochs=15
    if [ ! -f ${mexpdir}/model.loss.best ]; then
	mkdir -p $mexpdir/results; cp $multnn_dir/results/{model.acc.best,model.conf} $mexpdir/results
	${cuda_cmd} ${mexpdir}/train.log \
	    asr_train.py \
	    --ngpu ${ngpu} \
	    --backend ${backend} \
	    --outdir ${mexpdir}/results \
	    --debugmode ${debugmode} \
	    --dict ${dict} \
	    --debugdir ${mexpdir} \
	    --minibatches ${N} \
	    --verbose ${verbose} \
	    --resume ${multnn_resume} \
	    --train-json ${data_train}/data.json \
	    --valid-json ${data_dev}/data.json \
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
	    --batch-size 30 \
	    --maxlen-in ${maxlen_in} \
	    --maxlen-out ${maxlen_out} \
	    --opt sgd \
	    --lr 1e-4 \
	    --epochs ${epochs} \
	    --adapt yes \
	    --freeze yes \
	    --adaptLayerNames AttCtcOut \
	    --noencs_freeze $((elayers*2))
    fi
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2: NN fine tunning"
    
    epochs=30
    if [ ! -f ${m2expdir}/model.loss.best ]; then
	mkdir -p $m2expdir/results; cp $mexpdir/results/{model.acc.best,model.conf} $m2expdir/results
	${cuda_cmd} ${m2expdir}/train.log \
	    asr_train.py \
	    --ngpu ${ngpu} \
	    --backend ${backend} \
	    --outdir ${m2expdir}/results \
	    --debugmode ${debugmode} \
	    --dict ${dict} \
	    --debugdir ${m2expdir} \
	    --minibatches ${N} \
	    --verbose ${verbose} \
	    --resume ${resume} \
	    --train-json ${data_train}/data.json \
	    --valid-json ${data_dev}/data.json \
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
	    --batch-size 30 \
	    --maxlen-in ${maxlen_in} \
	    --maxlen-out ${maxlen_out} \
	    --opt sgd \
	    --lr 1e-2 \
	    --epochs ${epochs} \
	    --adapt yes 
    fi
fi

if [ ${stage} -le 3 ]; then
    echo "stage 7: Decoding"
    nj=60

    for data in ${data_eval}; do

	(
            expdir=$m2expdir
	    data_name=$(basename $data)
	    decode_dir=decode_${data_name}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
	    feat_recog_dir=${dumpdir}/${data}/$dumpdir_name
            # Copy data files for scoring and data split
            for f in utt2spk spk2utt text wav.scp segments; do
		[ ! -e ${feat_recog_dir}/${f} ] && cp data/$rtask/$f ${feat_recog_dir}/
            done
	    
            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
	    
            #### use CPU for decoding
            ngpu=0
	    
            # For rnnlm
            extra_opts=""
	    
	    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
		asr_recog.py \
		--ngpu ${ngpu} \
		--backend ${backend} \
		--recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
		--result-label ${expdir}/${decode_dir}/data.JOB.json \
		--model ${expdir}/results/model.${recog_model}  \
		--model-conf ${expdir}/results/model.conf  \
		--beam-size ${beam_size} \
		--penalty ${penalty} \
		--ctc-weight ${ctc_weight} \
		--maxlenratio ${maxlenratio} \
		--minlenratio ${minlenratio} \
		${extra_opts} &
	    
            score_sclite.sh --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}
	    
	) &
    done
    wait
    echo "Finished"
fi

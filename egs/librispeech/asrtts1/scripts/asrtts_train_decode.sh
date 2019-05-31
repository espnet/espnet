#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

# minibatch related
batchsize=20
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# other config
drop=0.2

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=10
patience=3

# rnnlm related
lm_layers=1
lm_units=1024
lm_opt=sgd        # or adam
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=1024 # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_patience=3
lm_maxlen=40      # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=0.7
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.5
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.
asr_train=true
asr_decode=true

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_speech_set=$1
train_text_set=$2
spk_vector=$3
train_paired_set=$4
train_unpaired_set=$5
train_dev=$6
recog_set=$7 #"test_clean test_other dev_clean dev_other"
expname=$8
unpair=$9
rnnlm_loss=$10

expdir=exp/${expname}
mkdir -p ${expdir}

feat_tr_paired_dir=dump/$train_paired_set/deltafalse
feat_tr_unpaired_dir=dump/$train_unpaired_set/deltafalse

if [ ! -s  ${feat_tr_unpaired_set}/data_rnd.json ]; then
    local/rand_datagen.sh --jsonout "data_rnd.json" --xvec ${spk_vector} dump/$train_speech_set/deltafalse dump/$train_text_set/deltafalse $feat_tr_unpaired_dir
fi

if [ $unpair == 'dual' ]; then
    tr_json_list="${feat_tr_unpaired_dir}/data_rnd.json"
elif [ $unpair == 'dualp' ]; then
   tr_json_list="${feat_tr_unpaired_dir}/data_rnd.json ${feat_tr_paired_dir}/data.json"
else
    tr_json_list="${feat_tr_paired_dir}/data.json"
fi


if [ "$policy_gradient" = "true" ]; then
    expdir=${expdir}_exploss_pgrad
    train_opts="$train_opts --policy-gradient"
fi
if [ "$freeze_encoder" = "true" ]; then
    expdir=${expdir}_freezeenc
    train_opts="$train_opts --freeze encatt"
fi
if [ "$rnnlm_loss" = "ce" ]; then
    expdir=${expdir}_rnnlmloss_${rnnlm_loss}
    train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss ce" 
elif [ "$rnnlm_loss" = "kld" ]; then
    expdir=${expdir}_rnnlmloss_${rnnlm_loss}
    train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss kld" 
elif [ "$rnnlm_loss" = "mmd" ]; then
    expdir=${expdir}_rnnlmloss_${rnnlm_loss}
    train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss mmd" 
elif [ "$rnnlm_loss" = "kl" ]; then
    expdir=${expdir}_rnnlmloss_${rnnlm_loss}
    train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss kl" 

fi

if [ $asrtts_train == 'true' ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
<<<<<<< HEAD
        asrtts_train.py \
=======
        asr_dual_train.py \
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${tr_json_list} \
        --valid-json ${feat_dt_dir}/data.json \
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
        --epochs ${epochs} \
        --asr-model-conf $asr_model_conf \
        --asr-model $asr_model \
        --tts-model-conf $tts_model_conf \
        --tts-model $tts_model \
        --expected-loss tts \
        --criterion acc \
        --update-asr-only \
        --generator tts \
        --sample-topk $sample_topk \
        --sample-scaling $sample_scaling \
        --teacher-weight $teacher_weight \
        --n-samples-per-input $n_samples \
        --adim 100 \
        --modify-output \
        $train_opts
fi

if [ $asrtts_decode == 'true' ]; then
    echo "stage 5: Decoding"
    nj=32
    for rtask in ${recog_set}; do
    (
        recog_opts=
        if [ $use_rnnlm = true ]; then
            decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
            recog_opts="$recog_opts --lm-weight ${lm_weight} --rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.json  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            $recog_opts \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

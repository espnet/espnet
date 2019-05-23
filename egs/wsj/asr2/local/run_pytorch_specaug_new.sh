#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh


# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# network architecture
ninit=pytorch

# encoder related
input_layer=conv2d     # encoder architecture type
elayers=12
eunits=2048
# decoder related
dlayers=6
dunits=2048
# attention related
adim=256
aheads=4

# hybrid CTC/attention
mtlalpha=0.3

# label smoothing
lsm_type=unigram
lsm_weight=0.1

# minibatch related
batchsize=32
maxlen_in=512  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
len_norm=False
opt=noam
epochs=100
lr_init=10.0
warmup_steps=25000
dropout=0.1
attn_dropout=0.0
accum_grad=2
grad_clip=5
patience=3

# decoding parameter
lm_weight=1.0
beam_size=10
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
n_decode_job=32
# scheduled sampling option
samp_prob=0.0

# args from run.sh (data prep + LM)
dumpdir=dump   # directory to dump full features
do_delta=false
train_set=train_si284
train_dev=test_dev93
recog_set="test_dev93 test_eval92"
dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
lmtag=1layer_unit1000_sgd_bs300
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
use_wordlm=true


# exp tag
tag="specaug_preprocess" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_transformer_${input_layer}_e${elayers}_unit${eunits}_d${dlayers}_unit${dunits}_aheads${aheads}_dim${adim}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_ngpu${ngpu}_bs${batchsize}_lr${lr_init}_warmup${warmup_steps}_mli${maxlen_in}_mlo${maxlen_out}_epochs${epochs}_accum${accum_grad}_lennorm${len_norm}
    if [ "${lsm_type}" != "" ]; then
        expname=${expname}_lsm${lsm_type}${lsm_weight}
    fi
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
        --accum-grad ${accum_grad} \
        --ngpu ${ngpu} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --aheads ${aheads} \
        --adim ${adim} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --dropout-rate ${dropout} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --grad-clip ${grad_clip} \
        --sampling-probability ${samp_prob} \
        --epochs ${epochs} \
        --sortagrad ${sortagrad} \
        --lsm-weight ${lsm_weight} \
        --preprocess-conf "conf/preprocess.json" \
        --model-module "espnet.nets.${backend}_backend.e2e_asr_transformer:E2E" \
        --transformer-lr ${lr_init} \
        --transformer-warmup-steps ${warmup_steps} \
        --transformer-input-layer ${input_layer} \
        --transformer-attn-dropout-rate ${attn_dropout} \
        --transformer-length-normalized-loss ${len_norm} \
        --transformer-init ${ninit} \
        --patience ${patience}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=${n_decode_job}
    if [ ${n_average} -gt 1 ]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        if [ ${lm_weight} == 0 ]; then
            recog_opts=""
        else
            decode_dir=${decode_dir}_rnnlm${lm_weight}_${lmtag}
            if [ ${use_wordlm} = true ]; then
                recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --batchsize 0 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            --verbose ${verbose} \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpu in training
nj=32        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)
# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length
# encoder related
embed_dim=512
elayers=1
eunits=512
econv_layers=3 # if set 0, no conv layer is used
econv_chans=512
econv_filts=5
# decoder related
dlayers=2
dunits=1024
prenet_layers=2  # if set 0, no prenet is used
prenet_units=256
postnet_layers=5 # if set 0, no postnet is used
postnet_chans=512
postnet_filts=5
use_speaker_embedding=true
# attention related
atype=forward_ta
adim=128
aconv_chans=32
aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
cumulate_att_w=true # whether to cumulate attetion weight
use_batch_norm=true # whether to use batch normalization in conv layer
use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
use_residual=false  # whether to use residual connection in encoder convolution
use_masking=true    # whether to mask the padded part in loss calculation
bce_pos_weight=1.0  # weight for positive samples of stop token in cross-entropy calculation
reduction_factor=2
# minibatch related
batchsize=64
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
batch_sort_key=output # empty or input or output (if empty, shuffled batch will be used)
maxlen_in=150     # if input length  > maxlen_in, batchsize is reduced (if batch_sort_key="", not effect)
maxlen_out=400    # if output length > maxlen_out, batchsize is reduced (if batch_sort_key="", not effect)
# optimization related
lr=1e-3
eps=1e-6
weight_decay=0.0
dropout=0.5
zoneout=0.1
epochs=30
patience=5
# decoding related
model=model.loss.best
threshold=0.5    # threshold to stop the generation
maxlenratio=10.0 # maximum length of generated samples = input length * maxlenratio
minlenratio=0.0  # minimum length of generated samples = input length * minlenratio

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=$1
dev_set=$2
eval_set=$3
expname=$4

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
dict=data/lang_1char/${train_set}_units.txt
expdir=exp/${expname}
mkdir -p ${expdir}
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --embed_dim ${embed_dim} \
           --elayers ${elayers} \
           --eunits ${eunits} \
           --econv_layers ${econv_layers} \
           --econv_chans ${econv_chans} \
           --econv_filts ${econv_filts} \
           --dlayers ${dlayers} \
           --dunits ${dunits} \
           --prenet_layers ${prenet_layers} \
           --prenet_units ${prenet_units} \
           --postnet_layers ${postnet_layers} \
           --postnet_chans ${postnet_chans} \
           --postnet_filts ${postnet_filts} \
           --atype ${atype} \
           --adim ${adim} \
           --aconv-chans ${aconv_chans} \
           --aconv-filts ${aconv_filts} \
           --cumulate_att_w ${cumulate_att_w} \
           --use_speaker_embedding ${use_speaker_embedding} \
           --use_batch_norm ${use_batch_norm} \
           --use_concate ${use_concate} \
           --use_residual ${use_residual} \
           --use_masking ${use_masking} \
           --bce_pos_weight ${bce_pos_weight} \
           --lr ${lr} \
           --eps ${eps} \
           --dropout ${dropout} \
           --zoneout ${zoneout} \
           --reduction_factor ${reduction_factor} \
           --weight-decay ${weight_decay} \
           --sortagrad ${sortagrad} \
           --batch_sort_key ${batch_sort_key} \
           --batch-size ${batchsize} \
           --maxlen-in ${maxlen_in} \
           --maxlen-out ${maxlen_out} \
           --epochs ${epochs} \
           --patience ${patience}

outdir=${expdir}/outputs_${model}_th${threshold}_mlr${minlenratio}-${maxlenratio}
    for name in ${dev_set} ${eval_set};do
        [ ! -e  ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --threshold ${threshold} \
                --maxlenratio ${maxlenratio} \
                --minlenratio ${minlenratio}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    done
    for name in ${dev_set} ${eval_set};do
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    done

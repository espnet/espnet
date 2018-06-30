#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# stage setting
stage=-1
# gpu setting
ngpu=1
# feature extraction related
fs=22050
fmax=""
fmin=""
n_mels=80
n_fft=1024
n_shift=512
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
# attention related
adim=128
aconv_chans=32
aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
cumulate_att_w=true # whether to cumulate attetion weight
use_batch_norm=true # whether to use batch normalization in conv layer
use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
use_residual=false  # whether to concatenate encoder embedding with decoder lstm outputs
use_masking=true    # whether to mask the padded part in loss calculation
bce_pos_weight=20.0
# minibatch related
batchsize=64
batch_sort_key="" # none or input or output
maxlen_in=150  # if input length  > maxlen_in, batchsize is reduced (if batch_sort_key="", not effect)
maxlen_out=400 # if output length > maxlen_out, batchsize is reduced (if batch_sort_key="", not effect)
epochs=200
# optimization related
lr=1e-3
eps=1e-6
weight_decay=0.0
dropout=0.5
zoneout=0.1
# decoding related
model=model.loss.best
threshold=0.5
maxlenratio=10.0
minlenratio=0.0
nj=32
# other
resume=""
verbose=0
seed=1125
dumpdir=dump   # directory to dump full features

# feature configuration
do_delta=false # true when using CNN

# root directory of db
db_root=downloads

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_no_dev
train_dev=train_dev
eval_set=eval

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    local/download.sh ${db_root}
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/LJSpeech-1.1 data/train
    utils/validate_data_dir.sh --no-feats data/train
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}/delta${do_delta}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=fbank
    local/make_fbank.sh --cmd "${train_cmd}" --nj 32 \
        --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
        --n_mels ${n_mels} --n_fft ${n_fft} --n_shift ${n_shift} \
        data/train exp/make_fbank/train ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --last data/train 500 data/deveval
    utils/subset_data_dir.sh --last data/deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/deveval 250 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/train ${n} data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi


if [ -z ${tag} ];then
    expdir=exp/${train_set}_taco2_enc${embed_dim}
    if [ ${econv_layers} -gt 0 ];then
        expdir=${expdir}-${econv_layers}x${econv_filts}x${econv_chans}
    fi
    expdir=${expdir}-${elayers}x${eunits}_dec${dlayers}x${dunits}
    if [ ${prenet_layers} -gt 0 ];then
        expdir=${expdir}_pre${prenet_layers}x${prenet_units}
    fi
    if [ ${postnet_layers} -gt 0 ];then
        expdir=${expdir}_post${postnet_layers}x${postnet_filts}x${postnet_chans}
    fi
    expdir=${expdir}_att${adim}-${aconv_filts}x${aconv_chans}
    if ${cumulate_att_w};then
        expdir=${expdir}_cm
    fi
    if ${use_batch_norm};then
        expdir=${expdir}_bn
    fi
    if ${use_residual};then
        expdir=${expdir}_rs
    fi
    if ${use_concate};then
        expdir=${expdir}_cc
    fi
    if ${use_masking};then
        expdir=${expdir}_msk_pw${bce_pos_weight}
    fi
    expdir=${expdir}_do${dropout}_zo${zoneout}_lr${lr}_ep${eps}_wd${weight_decay}_bs$((batchsize*ngpu))
    if [ ! -z ${batch_sort_key} ];then
        expdir=${expdir}_sort_by_${batch_sort_key}_mli${maxlen_in}_mlo${maxlen_out}
    fi
    expdir=${expdir}_sd${seed}
else
    expdir=exp/${train_set}_${tag}
fi
if [ ${stage} -le 3 ];then
    echo "stage 3: Back translator training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        bts_train.py \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
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
           --adim ${adim} \
           --aconv-chans ${aconv_chans} \
           --aconv-filts ${aconv_filts} \
           --cumulate_att_w ${cumulate_att_w} \
           --use_batch_norm ${use_batch_norm} \
           --use_concate ${use_concate} \
           --use_residual ${use_residual} \
           --use_masking ${use_masking} \
           --bce_pos_weight ${bce_pos_weight} \
           --lr ${lr} \
           --eps ${eps} \
           --dropout-rate ${dropout} \
           --zoneout-rate ${zoneout} \
           --weight-decay ${weight_decay} \
           --batch_sort_key ${batch_sort_key} \
           --batch-size ${batchsize} \
           --maxlen-in ${maxlen_in} \
           --maxlen-out ${maxlen_out} \
           --epochs ${epochs}
fi

outdir=${expdir}/outputs_${model}_th${threshold}_mlr${minlenratio}-${maxlenratio}
if [ ${stage} -le 4 ];then
    echo "stage 4: Decoding"
    for sets in ${train_dev} ${eval_set};do
        [ ! -e  ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
        splitjson.py --parts ${nj} ${dumpdir}/${sets}/delta${do_delta}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:$nj ${outdir}/${sets}/log/decode.JOB.log \
            bts_decode.py \
                --backend pytorch \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${sets}/feats.JOB \
                --json ${dumpdir}/${sets}/delta${do_delta}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --model-conf ${expdir}/results/model.conf \
                --threshold ${threshold} \
                --maxlenratio ${maxlenratio} \
                --minlenratio ${minlenratio}
        # concatenate scp files
        for n in $(seq $nj); do
            cat "${outdir}/${sets}/feats.$n.scp" || exit 1;
        done > ${outdir}/${sets}/feats.scp
        # remove temp json
        rm ${outdir}/${sets}/*.json 2>/dev/null
    done
fi

if [ ${stage} -le 5 ];then
    echo "stage 5: Synthesis"
    for sets in ${train_dev} ${eval_set};do
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        local/convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
            --n_mels ${n_mels} --n_fft ${n_fft} --n_shift ${n_shift} \
            ${outdir}_denorm/${sets} \
            ${outdir}_denorm/${sets}/log \
            ${outdir}_denorm/${sets}/wav
    done
fi

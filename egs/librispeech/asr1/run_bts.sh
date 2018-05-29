#!/bin/bash

. ./path.sh
. ./cmd.sh

# stage setting
stage=5
# gpu setting
ngpu=1
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
prenet_layers=2 # if set 0, no prenet is used
prenet_units=256
postnet_layers=5 # if set 0, no postnet is used
postnet_chans=512
postnet_filts=5
# attention related
adim=512
aconv_chans=32
aconv_filts=15 # resulting in filter_size = aconv_filts * 2 + 1
cumulate_att_w=true # whether to cumulate attetion weight
use_batch_norm=true # whether to use batch normalization in conv layer
use_concate=true # whether to concatenate encoder embedding with decoder lstm outputs
use_residual=false # whether to concatenate encoder embedding with decoder lstm outputs
use_masking=false
bce_pos_weight=20.0
# minibatch related
batchsize=32
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced
epochs=30
# optimization related
lr=1e-3
eps=1e-6
weight_decay=0.0
# other
do_delta=false
target=states # feats or states
train_set=train_360
train_dev=dev
decode_set="train_100 train_other_500"
verbose=1
tag=
# decoding related
threshold=0.5
maxlenratio=5.0
minlenratio=0.0
nj=64
# decoder retraining related
flatstart=false

. utils/parse_options.sh
set -e

basedir=exp/${train_set}_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150
dumpdir=${basedir}/outputs
model=${basedir}/results/model.acc.best
config=${basedir}/results/model.conf
dict=data/lang_1char/${train_set}_units.txt

if [ ${stage} -le 5 ];then
    echo "stage 5: Encoder state extraction"
    for sets in ${train_set} ${train_dev};do
        [ ! -e ${dumpdir}/${sets} ] && mkdir -p ${dumpdir}/${sets}
        # split scp file
        scp=dump/${sets}/delta${do_delta}/feats.scp
        split_scps=""
        for n in $(seq $nj); do
            split_scps="$split_scps ${dumpdir}/${sets}/log/feats.$n.scp"
        done
        utils/split_scp.pl $scp $split_scps || exit 1;
        # decode
        ${train_cmd} JOB=1:$nj ${dumpdir}/${sets}/log/extract.log \
            asr_extract.py \
                --backend pytorch \
                --ngpu 0 \
                --out ${dumpdir}/${sets}/feats.JOB \
                --feat ${dumpdir}/${sets}/log/feats.JOB.scp \
                --model ${model} \
                --model-conf ${config}
        # concatenate scp files
        for n in $(seq $nj); do
            cat ${dumpdir}/${sets}/feats.$n.scp || exit 1;
        done > ${dumpdir}/${sets}/feats.scp
        # remove temp scps
        rm ${dumpdir}/${sets}/log/feats.*.scp 2>/dev/null
        # create json
        data2json.sh --feat ${dumpdir}/${sets}/feats.scp \
             data/${sets} ${dict} > ${dumpdir}/${sets}/data.json
    done
fi

if [ -z ${tag} ];then
    expdir=exp/${train_set}_tacotron2_${target}_enc${embed_dim}
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
        expdir=${expdir}_cumulate
    fi
    if ${use_batch_norm};then
        expdir=${expdir}_bn
    fi
    if ${use_residual};then
        expdir=${expdir}_res
    fi
    if ${use_concate};then
        expdir=${expdir}_concate
    fi
    if ${use_masking};then
        expdir=${expdir}_masking_pw${bce_pos_weight}
    fi
    expdir=${expdir}_lr${lr}_ep${eps}_wd${weight_decay}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
else
    expdir=exp/${train_set}_${tag}
fi
if [ ${stage} -le 6 ];then
    echo "stage 6: Back translator training"
    if [ ${target} == "states" ];then
        tr_feat=scp:${dumpdir}/${train_set}/enc_hs/feats.scp
        tr_label=${dumpdir}/${train_set}/enc_hs/data.json
        dt_feat=scp:${dumpdir}/${train_dev}/enc_hs/feats.scp
        dt_label=${dumpdir}/${train_dev}/enc_hs/data.json
    else
        tr_feat=scp:dump/${train_set}/delta${do_delta}/feats.scp
        tr_label=dump/${train_set}/delta${do_delta}/data.json
        dt_feat=scp:dump/${train_dev}/delta${do_delta}/feats.scp
        dt_label=dump/${train_dev}/delta${do_delta}/data.json
    fi
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        bts_train.py \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --verbose ${verbose} \
           --train-feat ${tr_feat} \
           --train-label ${tr_label} \
           --valid-feat ${dt_feat} \
           --valid-label ${dt_label} \
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
           --weight-decay ${weight_decay} \
           --batch-size ${batchsize} \
           --maxlen-in ${maxlen_in} \
           --maxlen-out ${maxlen_out} \
           --epochs ${epochs}
fi

outdir=${expdir}/outputs_th${threshold}_mlr${minlenratio}-${maxlenratio}
if [ ${stage} -le 7 ];then
    echo "stage 7: Decoding"
    for sets in ${decode_set};do
        [ ! -e  ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
        # create split json
        data2json.sh data/${sets} ${dict} > ${outdir}/${sets}/data.json
        splitjson.py -n $nj ${outdir}/${sets}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:$nj ${outdir}/${sets}/log/decode.JOB.log \
            bts_decode.py \
                --backend pytorch \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${sets}/feats.JOB \
                --label ${outdir}/${sets}/data.JOB.json \
                --model ${expdir}/results/model.loss.best \
                --model-conf ${expdir}/results/model.conf \
                --threshold ${threshold} \
                --maxlenratio ${maxlenratio} \
                --minlenratio ${minlenratio}
        # concatenate scp files
        for n in $(seq $nj); do
            cat ${outdir}/${sets}/feats.$n.scp || exit 1;
        done > ${outdir}/${sets}/feats.scp
        # remove temp json
        rm ${outdir}/${sets}/*.json 2>/dev/null
    done
fi

if [ ${stage} -le 8 ];then
    echo "stage 8: Re-training decoder"
    retroutdir=exp/re${train_set}_bs${batchsize}_mli${maxlen_out}_mlo${maxlen_in}
    if ${flatstart};then
        retroutdir=${retroutdir}_flatstart
    fi
    if [ ! -e ${retroutdir}/data/train/data.json ];then
        # copy train data with ground-truth scp
        utils/copy_data_dir.sh data/${train_set} ${retroutdir}/data/${train_set}
        cp ${dumpdir}/${train_set}/feats.scp ${retroutdir}/data/${train_set}
        # copy decode data with generated scp
        for sets in ${decode_set};do
            utils/copy_data_dir.sh data/${sets} ${retroutdir}/data/${sets}
            cp ${outdir}/${sets}/feats.scp ${retroutdir}/data/${sets}
        done
        # combine train and decode data
        combdirs=
        for sets in ${train_set} ${decode_set};do
            combdirs="$combdirs ${retroutdir}/data/${sets}"
        done
        utils/combine_data.sh ${retroutdir}/data/train ${combdirs}
        # create json
        data2json.sh --feat ${retroutdir}/data/train/feats.scp \
             ${retroutdir}/data/train ${dict} > ${retroutdir}/data/train/data.json
    fi
    # re-train
    ${cuda_cmd} --gpu ${ngpu} ${retroutdir}/retrain.log \
        asr_retrain.py \
            --ngpu ${ngpu} \
            --model ${model} \
            --model-conf ${config} \
            --verbose ${verbose} \
            --outdir ${retroutdir}/results \
            --dict ${dict} \
            --train-feat scp:${retroutdir}/data/train/feats.scp \
            --train-label ${retroutdir}/data/train/data.json \
            --valid-feat scp:dump/${train_dev}/delta${do_delta}/feats.scp \
            --valid-label dump/${train_dev}/delta${do_delta}/data.json \
            --batch-size ${batchsize} \
            --maxlen-in ${maxlen_out} \
            --maxlen-out ${maxlen_in} \
            --epochs ${epochs} \
            --flatstart ${flatstart}
fi

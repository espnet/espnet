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
econv_layers=3
econv_chans=512
econv_filts=5
# decoder related
dlayers=2
dunits=1024
prenet_layers=2
prenet_units=256
postnet_layers=5
postnet_chans=512
postnet_filts=5
# attention related
adim=512
aconv_chans=32
aconv_filts=15 # resulting in 31
cumulate_att_w=true
# minibatch related
batchsize=32
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced
epochs=30
# optimization related
lr=1e-3
eps=1e-6
# other
do_delta=false
target=states # feats or states
train_set=train_100
train_dev=dev
verbose=1
tag=

. utils/parse_options.sh
set -e

dumpdir=./exp/${train_set}_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150/outputs
model=./exp/${train_set}_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150/results/model.acc.best
config=./exp/${train_set}_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150/results/model.conf

if [ ${stage} -le 5 ];then
    echo "stage 5: Encoder state extraction"
    dict=./data/lang_1char/${train_set}_units.txt
    for sets in ${train_set} ${train_dev};do
        featdir=dump/${sets}/delta${do_delta}
        ${cuda_cmd} --gpu ${ngpu} ${dumpdir}/log/extract.log \
            asr_extract.py \
                --backend pytorch \
                --ngpu ${ngpu} \
                --outdir ${dumpdir}/${sets}/tmp \
                --feat scp:${featdir}/feats.scp \
                --label ${featdir}/data.json \
                --model ${model} \
                --model-conf ${config}
        [ ! -e ${dumpdir}/${sets}/enc_hs ] && mkdir -p ${dumpdir}/${sets}/enc_hs
        copy-feats ark:${dumpdir}/${sets}/tmp/feats.ark \
            ark,scp:${dumpdir}/${sets}/enc_hs/feats.ark,${dumpdir}/${sets}/enc_hs/feats.scp
        data2json.sh --feat ${dumpdir}/${sets}/enc_hs/feats.scp \
             data/${sets} ${dict} > ${dumpdir}/${sets}/enc_hs/data.json
        [ -e ${dumpdir}/${sets}/tmp ] && rm -rf ${dumpdir}/${sets}/tmp
    done
fi

if [ -z ${tag} ];then
    expdir=exp/${train_set}_tacotron2_${target}_enc${embed_dim}-${econv_layers}x${econv_filts}x${econv_chans}-${elayers}x${eunits}_dec${dlayers}x${dunits}_pre${prenet_layers}x${prenet_units}_post${postnet_layers}x${postnet_filts}x${postnet_chans}_att${adim}-${aconv_filts}x${aconv_chans}_lr${lr}_eps${eps}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${cumulate_att_w};then
        expdir=${expdir}_cumulate
    fi
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
           --lr ${lr} \
           --eps ${eps} \
           --batch-size ${batchsize} \
           --maxlen-in ${maxlen_in} \
           --maxlen-out ${maxlen_out} \
           --epochs ${epochs}
fi

# if [ ${stage} -le 7 ];then
#     echo "stage 7: Decoding"
#     dict=./data/lang_1char/${train_set}_units.txt
#     for sets in ${train_set} ${train_dev};do
#         featdir=${dumpdir}/${sets}/enc_hs
#         ${cuda_cmd} --gpu ${ngpu} ${expdir}/outputs/${sets}/log/decode.log \
#             bts_decode.py \
#                 --backend pytorch \
#                 --ngpu ${ngpu} \
#                 --outdir ${expdir}/outputs/${sets} \
#                 --feat scp:${featdir}/feats.scp \
#                 --label ${featdir}/data.json \
#                 --model ${expdir}/results/model.loss.best \
#                 --model-conf ${expdir}/results/model.conf
#     done
# fi

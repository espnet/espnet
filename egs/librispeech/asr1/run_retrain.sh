#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# stage setting
stage=5
# gpu setting
ngpu=1
batchsize=50
maxlen_in=400  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=200 # if output length > maxlen_out, batchsize is automatically reduced
epochs=15
# other
do_delta=false
train_set=train_100
train_dev=dev
decode_set="train_360 train_other_500"
verbose=0
tag=
nj=32
# decoder retraining related
input_layer_idx=-1
flatstart=false
freeze_attention=false
# decoding related
beam_size=20
penalty=0.0
maxlenratio=0.8
minlenratio=0.3
ctc_weight=0.0
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'
recog_set="test_clean test_other dev_clean dev_other"

. utils/parse_options.sh
set -e

basedir=exp/${train_set}_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150
dumpdir=${basedir}/outputs-h${input_layer_idx}
model=${basedir}/results/model.acc.best
config=${basedir}/results/model.conf
dict=data/lang_1char/${train_set}_units.txt
outdir=./exp/train_100_taco2_states_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_att128-15x32_cm_bn_cc_msk_pw20.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs50_sort_by_input_mli150_mlo400/outputs_th0.5_mlr0.0-5.0

retroutdir=$(dirname ${outdir})/re${train_set}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
if ${flatstart};then
    retroutdir=${retroutdir}_fs
fi
if ${freeze_attention};then
    retroutdir=${retroutdir}_fz_att
fi
if [ ${stage} -le 8 ];then
    echo "stage 8: Re-training decoder"
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
            --train-json ${retroutdir}/data/train/data.json \
            --valid-json dump/${train_dev}/delta${do_delta}/data.json \
            --batch-size ${batchsize} \
            --maxlen-in ${maxlen_out} \
            --maxlen-out ${maxlen_in} \
            --epochs ${epochs} \
            --freeze-attention ${freeze_attention} \
            --flatstart ${flatstart} \
            --input-layer-idx ${input_layer_idx}
fi

if [ ${stage} -le 9 ]; then
    echo "stage 9: Decoding"
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

         # make json labels for recognition
        for j in `seq 1 ${nj}`; do
            data2json.sh --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} \
                ${sdata}/${j} ${dict} > ${sdata}/${j}/data.json
        done

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${retroutdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend pytorch \
            --recog-json ${sdata}/JOB/data.json \
            --result-label ${retroutdir}/${decode_dir}/data.JOB.json \
            --model ${retroutdir}/results/model.${recog_model}  \
            --model-conf ${config} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            &
        wait

        score_sclite.sh --wer true ${retroutdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

#!/bin/bash

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0
resume=        # Resume the training from snapshot
nj=16

# feature configuration
do_delta=false

# configs
## Change training config to conf/tuning/train_rnnt_att.yaml
## if you want to use RNN-T with attention
train_config=conf/tuning/train_rnnt.yaml
decode_config=conf/tuning/decode_rnnt.yaml

# decoding parameter
recog_model=model.loss.best
batchsize=0 # batchsize during decoding. > 1 is not support yet 

# lm parameter
lm_config=conf/tuning/lm.yaml
lm_weight=0.1
use_lm=true

# data
voxforge=downloads # original data directory to be stored
lang=it # de, en, es, fr, it, nl, pt, ru

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_${lang}
train_dev=dt_${lang}
recog_set="dt_${lang} et_${lang}"
lm_test="et_${lang}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"

    local/getdata.sh ${lang} ${voxforge}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"

    selected=${voxforge}/${lang}/extracted

    local/voxforge_data_prep.sh ${selected} ${lang}
    local/voxforge_format_data.sh ${lang}
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    fbankdir=fbank

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
        data/all_${lang} exp/make_fbank/train_${lang} ${fbankdir}
    utils/fix_data_dir.sh data/all_${lang}

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or 0 characters
    remove_longshortdata.sh data/all_${lang} data/all_trim_${lang}

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_trim_${lang} data/tr_${lang} data/dt_${lang} data/et_${lang}
    rm -r data/all_trim_${lang}

    # compute global CMVN
    compute-cmvn-stats scp:data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj $nj --do_delta ${do_delta} \
        data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj $nj --do_delta ${do_delta} \
        data/dt_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        
        dump.sh --cmd "$train_cmd" --nj $nj --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/tr_${lang}_units.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict}
    text2token.py -s 1 -n 1 data/tr_${lang}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --lang ${lang} --feat ${feat_tr_dir}/feats.scp \
         data/tr_${lang} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --lang ${lang} --feat ${feat_dt_dir}/feats.scp \
         data/dt_${lang} ${dict} > ${feat_dt_dir}/data.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

lmexpname=train_rnnlm_${backend}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"

    lmdatadir=data/local/lm_train
    lmdict=${dict}
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1 data/${train_set}/text \
        | cut -f 2- -d" " > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text \
        | cut -f 2- -d" " > ${lmdatadir}/valid.txt
    text2token.py -s 1 -n 1 data/${lm_test}/text \
        | cut -f 2- -d" " > ${lmdatadir}/test.txt

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. single gpu will be used."
    fi

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
                lm_train.py \
                --config ${lm_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --verbose 1 \
                --outdir ${lmexpdir} \
                --tensorboard-dir tensorboard/${lmexpname} \
                --train-label ${lmdatadir}/train.txt \
                --valid-label ${lmdatadir}/valid.txt \
                --test-label ${lmdatadir}/test.txt \
                --dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
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
        asr_rnnt_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

lm_model=${lmexpdir}/rnnlm.model.best
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"

    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        if [ "${use_lm}" = true ]; then
            recog_opts="--rnnlm ${lm_model} --lm-weight ${lm_weight}"
        else
            recog_opts=""
        fi

        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_rnnt_recog.py \
            --config ${decode_config} \
            --batchsize ${batchsize} \
            --ngpu 0 \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            ${recog_opts}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
    done

    echo "Finished"
fi

#!/usr/bin/env bash

# Copyright 2019 Shun Kiyono
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# based on the wonderful work done by Hirofumi Inaguma

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of NMT models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best NMT models will be averaged.
                             # if false, the last `n_average` NMT models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
iwslt16=iwslt16_data

# target language related
src_lang=en
tgt_lang=de
# (kiyono) currently de only

nbpe=16000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -x
set -e
set -u
set -o pipefail

train_set=train.${src_lang}-${tgt_lang}.${tgt_lang}
train_dev=tst2012.${src_lang}-${tgt_lang}.${tgt_lang}
trans_set="tst2013.${src_lang}-${tgt_lang}.${tgt_lang} tst2014.${src_lang}-${tgt_lang}.${tgt_lang}"

mkdir -p ${dumpdir}/$train_set
mkdir -p ${dumpdir}/$train_dev
for dir in ${trans_set}; do
    mkdir -p ${dumpdir}/${dir}
done

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${iwslt16} ${tgt_lang}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/data_prep.sh ${iwslt16} de
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # 1. moses tokenization
    local/tokenize.sh ${iwslt16} ${src_lang} ${tgt_lang} ${dumpdir} ${train_set} ${train_dev} "$(echo ${trans_set} | tr ' ' '_')"

    # 2. moses true-casing
    local/truecasing.sh ${iwslt16} ${src_lang} ${tgt_lang} ${dumpdir} ${train_set} ${train_dev} "$(echo ${trans_set} | tr ' ' '_')"

    # 3. clean corpus
    local/clean_corpus.sh ${iwslt16} ${src_lang} ${tgt_lang} ${dumpdir} ${train_set}

    # 4. bpe training & splitting
    local/train_and_apply_bpe.sh ${iwslt16} ${src_lang} ${tgt_lang} ${dumpdir} ${train_set} ${train_dev} "$(echo ${trans_set} | tr ' ' '_')" ${nbpe}
fi

src_dict=${dumpdir}/vocab/vocab.${src_lang}
tgt_dict=${dumpdir}/vocab/vocab.${tgt_lang}
echo "source dictionary: ${src_dict}"
echo "target dictionary: ${tgt_dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p ${dumpdir}/vocab

    echo "Make vocabulary files"
    local/generate_vocab.py --input ${dumpdir}/${train_set}/train.tkn.tc.clean.${src_lang}_bpe${nbpe} > ${src_dict}
    local/generate_vocab.py --input ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang}_bpe${nbpe} > ${tgt_dict}

    echo "Make json files"
    local/generate_json.sh ${iwslt16} ${src_lang} ${tgt_lang} ${dumpdir} ${train_set} ${train_dev} "$(echo ${trans_set} | tr ' ' '_')" ${nbpe}
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        mt_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${tgt_dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${dumpdir}/${train_set}/data.json \
        --valid-json ${dumpdir}/${train_dev}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average NMT models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi
    nj=12

    pids=() # initialize pids
    for ttask in ${trans_set}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        # decoding complete --> evaluation
        local/compute_bleu.sh ${expdir}/${decode_dir} ${src_lang} ${tgt_lang} $ttask $feat_trans_dir

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

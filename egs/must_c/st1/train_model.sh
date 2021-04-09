#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16           # number of parallel jobs for decoding
debugmode=1
dumpdir=    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=10                  # the number of ST models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# pre-training related
asr_model=
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default


# target language related
#tgt_lang=de
#tgt_lang="de_es_fr_it_nl_pt_ro_ru"
tgt_lang=
#tgt_lang="it"
# you can choose from de, es, fr, it, nl, pt, ro, ru
# if you want to train the multilingual model, segment languages with _ as follows:
# e.g., tgt_lang="de_es_fr"
# if you want to use all languages, set tgt_lang="all"

# bpemode (unigram or bpe)
nbpe=8000
bpemode=bpe

asr_rnnlm=

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.en-${tgt_lang}.${tgt_lang}
#train_dev="dev.en-${tgt_lang}.${tgt_lang} tst-HE.en-${tgt_lang}.${tgt_lang}"
train_dev=dev.en-${tgt_lang}.${tgt_lang}
#trans_set=""
#for lang in $(echo ${tgt_lang} | tr '_' ' '); do
#    trans_set="${trans_set} tst-COMMON.en-${lang}.${lang} tst-HE.en-${lang}.${lang}"
#done
trans_set=tst-COMMON.en-${tgt_lang}.${tgt_lang}

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}

# NOTE: skip stage 3: LM Preparation
expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}_${tag}_gpu${ngpu}
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --config ${train_config} \
        --n-iter-process 8 \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Averaging Models"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
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
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
        else
            trans_model=model.last${n_average}.avg.best
        fi
    fi
    echo "using model: ${trans_model}"
    pids=() # initialize pids
    for ttask in ${train_dev}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir}/asr

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        #### use CPU for decoding
        ngpu=0

        if [[ -f "${asr_rnnlm}" ]]; then
            echo "using lm: ${asr_rnnlm}"
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                st_trans.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --asr-result-label ${expdir}/${decode_dir}/asr/data.JOB.json \
                --model ${expdir}/results/${trans_model} \
                --asr-rnnlm ${asr_rnnlm}
        else
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                st_trans.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --asr-result-label ${expdir}/${decode_dir}/asr/data.JOB.json \
                --model ${expdir}/results/${trans_model}
        fi

        score_bleu.sh --case ${tgt_case} --set ${ttask} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
            --remove_nonverbal ${remove_nonverbal} \
            ${expdir}/${decode_dir} ${tgt_lang} ${dict}

        local/score_sclite.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
            ${expdir}/${decode_dir}/asr ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
        else
            trans_model=model.last${n_average}.avg.best
        fi
    fi
    echo "using model: ${trans_model}"
    pids=() # initialize pids
    for ttask in ${trans_set}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir}/asr

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        #### use CPU for decoding
        ngpu=0

        if [[ -f "${asr_rnnlm}" ]]; then
            echo "using lm: ${asr_rnnlm}"
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                [ -f ${expdir}/${decode_dir}/data.JOB.json ] && st_trans.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --asr-result-label ${expdir}/${decode_dir}/asr/data.JOB.json \
                --model ${expdir}/results/${trans_model} \
                --asr-rnnlm ${asr_rnnlm} || echo "${expdir}/${decode_dir}/data.JOB.json already exists"
        else
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                st_trans.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --asr-result-label ${expdir}/${decode_dir}/asr/data.JOB.json \
                --model ${expdir}/results/${trans_model}
        fi

        score_bleu.sh --case ${tgt_case} --set ${ttask} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
            --remove_nonverbal ${remove_nonverbal} \
            ${expdir}/${decode_dir} ${tgt_lang} ${dict}

        local/score_sclite.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
            ${expdir}/${decode_dir}/asr ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
#if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#    echo "stage 4: Network Training"
#
#    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
#        st_train.py \
#        --config ${train_config} \
#        --preprocess-conf ${preprocess_config} \
#        --ngpu ${ngpu} \
#        --backend ${backend} \
#        --outdir ${expdir}/results \
#        --tensorboard-dir tensorboard/${expname} \
#        --debugmode ${debugmode} \
#        --dict ${dict} \
#        --debugdir ${expdir} \
#        --minibatches ${N} \
#        --seed ${seed} \
#        --verbose ${verbose} \
#        --resume ${resume} \
#        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
#        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
#        --enc-init ${asr_model} \
#        --dec-init ${mt_model}
#fi
#
#if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#    echo "stage 5: Decoding"
#    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
#        # Average ST models
#        if ${use_valbest_average}; then
#            trans_model=model.val${n_average}.avg.best
#            opt="--log ${expdir}/results/log --metric ${metric}"
#        else
#            trans_model=model.last${n_average}.avg.best
#            opt="--log"
#        fi
#        average_checkpoints.py \
#            ${opt} \
#            --backend ${backend} \
#            --snapshots ${expdir}/results/snapshot.ep.* \
#            --out ${expdir}/results/${trans_model} \
#            --num ${n_average}
#    fi
#
#    pids=() # initialize pids
#    for ttask in ${trans_set}; do
#    (
#        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
#        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
#
#        # split data
#        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
#
#        #### use CPU for decoding
#        ngpu=0
#
#        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
#            st_trans.py \
#            --config ${decode_config} \
#            --ngpu ${ngpu} \
#            --backend ${backend} \
#            --batchsize 0 \
#            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
#            --result-label ${expdir}/${decode_dir}/data.JOB.json \
#            --model ${expdir}/results/${trans_model}
#
#        score_bleu.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
#            --remove_nonverbal ${remove_nonverbal} \
#            ${expdir}/${decode_dir} ${tgt_lang} ${dict}
#    ) &
#    pids+=($!) # store background pids
#    done
#    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
#    echo "Finished"
#fi

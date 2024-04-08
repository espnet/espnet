#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1          # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# preprocessing related
src_case=lc.rm
tgt_case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train.en
train_dev=dev.en
recog_set=test.jp

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_data.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    for x in train dev test; do
        mkdir -p data/${x}/temp
        cut -f1 -d$'\t' db/split/${x} > data/${x}/temp/text.en
        cut -f2 -d$'\t' db/split/${x} > data/${x}/temp/text.jp
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    mkdir -p data/temp/${train_set}
    mkdir -p data/temp/${train_dev}
    mkdir -p data/temp/${recog_set}
    for lang in en jp; do
        cat data/train/temp/text.${lang} >  data/temp/${train_set}/${lang}.org
        cat data/dev/temp/text.${lang} >  data/temp/${train_dev}/${lang}.org
        cat data/test/temp/text.${lang} >  data/temp/${recog_set}/${lang}.org
    done

    for x in ${train_set} ${train_dev} ${recog_set}; do
        for lang in en jp; do
            dst=data/temp/${x}
            # normalize punctuation
            normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

            # lowercasing
            lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
            cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

            # remove punctuation
            local/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm.prev

            # Fill empty spaces
            < ${dst}/${lang}.norm.lc.rm.prev awk '{if(length($0)==0) {print "<NOISE>"} else print $0 }' > ${dst}/${lang}.norm.lc.rm.org

            # apply name
            < ${dst}/${lang}.norm.lc.rm.org awk '{print toupper("'${x}'-")NR " " $0}' >  ${dst}/${lang}.norm.lc.rm

            mkdir -p data/${x}.${lang}
            cp ${dst}/${lang}.norm.lc.rm data/${x}.${lang}/text.lc.rm
            mkdir -p data/${x}
            cp data/${x}.${lang}/text.lc.rm data/${x}/text.lc.rm.${lang}
        done
    done
fi

dict_src=data/lang_1char/${train_set}_units_${src_case}.en.txt
dict_tgt=data/lang_1char/${train_set}_units_${tgt_case}.jp.txt
nlsyms=data/lang_1char/non_lang_syms_${tgt_case}.txt
echo "dictionary (src): ${dict_src}"
echo "dictionary (tgt): ${dict_tgt}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- -d' ' data/${train_set}.*/text.${tgt_case} | grep "<" | grep ">"| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a target dictionary"
    echo "<unk> 1" > ${dict_tgt} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}.jp/text.${tgt_case} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict_tgt}
    wc -l ${dict_tgt}

    echo "make a source dictionary"
    echo "<unk> 1" > ${dict_src} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}.en/text.${src_case} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict_src}
    wc -l ${dict_src}

    echo "make json files"
    local/data2json.sh --nj 16 --text data/${train_set}.jp/text.${tgt_case} --nlsyms ${nlsyms} --skip_utt2spk true \
        data/${train_set} ${dict_tgt} > ${feat_tr_dir}/data.${src_case}_${tgt_case}.json
    local/data2json.sh --text data/${train_dev}.jp/text.${tgt_case} --nlsyms ${nlsyms} --skip_utt2spk true \
        data/${train_dev} ${dict_tgt} > ${feat_dt_dir}/data.${src_case}_${tgt_case}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/data2json.sh --text data/${rtask}.jp/text.${tgt_case} --nlsyms ${nlsyms}  --skip_utt2spk true \
            data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data.${src_case}_${tgt_case}.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev} ${recog_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/${x}.en
        local/update_json.sh --text ${data_dir}/text.${src_case} --nlsyms ${nlsyms} \
            ${feat_dir}/data.${src_case}_${tgt_case}.json ${data_dir} ${dict_src}
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${src_case}_${tgt_case}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        mt_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict_tgt} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data.${src_case}_${tgt_case}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.${src_case}_${tgt_case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        local/score_bleu.sh --case ${tgt_case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} jp ${dict_tgt}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

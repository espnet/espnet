#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

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
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# preprocessing related
src_case=lc.rm
tgt_case=lc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
datadir=/n/sd3/inaguma/corpus/libri_french/data
# libri_french
#  |_ train/
#  |_ other/
#  |_ dev/
#  |_ test/
# Download data from here:
# https://persyval-platform.univ-grenoble-alpes.fr/DS91/detaildataset

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

train_set=train.fr
train_set_prefix=train
train_dev=train_dev.fr
recog_set="dev.fr test.fr"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    for x in dev test train other; do
        local/data_prep.sh ${datadir}/${x} data/${x}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Divide into En Fr, Fr (google trans)
    for x in ${train_set_prefix} dev test; do
        local/divide_lang.sh ${x}
    done

    for lang in en fr fr.gtranslate; do
        if [ -d data/train_dev.${lang} ];then
            rm -rf data/train_dev.${lang}
        fi
        cp -rf data/dev.${lang} data/train_dev.${lang}
    done

    for x in ${train_set_prefix} train_dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in en fr fr.gtranslate; do
            remove_longshortdata.sh --no_feat true --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f -1 -d " " data/${x}.en.tmp/text > data/${x}.fr.tmp/reclist1
        cut -f -1 -d " " data/${x}.fr.tmp/text > data/${x}.fr.tmp/reclist2
        cut -f -1 -d " " data/${x}.fr.gtranslate.tmp/text > data/${x}.fr.tmp/reclist3
        comm -12 data/${x}.fr.tmp/reclist1 data/${x}.fr.tmp/reclist2 > data/${x}.fr.tmp/reclist4
        comm -12 data/${x}.fr.tmp/reclist3 data/${x}.fr.tmp/reclist4 > data/${x}.fr.tmp/reclist

        for lang in en fr fr.gtranslate; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.fr.tmp/reclist data/${x}.${lang}
            utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lang}
        done
        rm -rf data/${x}.*.tmp
    done
fi

dict_src=data/lang_1char/${train_set}_units_${src_case}.txt
dict_tgt=data/lang_1char/${train_set}_units_${tgt_case}.txt
nlsyms=data/lang_1char/non_lang_syms_${tgt_case}.txt
echo "dictionary (src): ${dict_src}"
echo "dictionary (tgt): ${dict_tgt}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- -d' ' data/${train_set_prefix}.*/text.${tgt_case} | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a target dictionary"
    echo "<unk> 1" > ${dict_tgt} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/${train_set_prefix}.*/text.${tgt_case} | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict_tgt}
    wc -l ${dict_tgt}

    echo "make a source dictionary"
    echo "<unk> 1" > ${dict_src} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/${train_set_prefix}.*/text.${src_case} | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict_src}
    wc -l ${dict_src}

    echo "make json files"
    local/data2json.sh --nj 16 --text data/${train_set}/text.${tgt_case} --nlsyms ${nlsyms} \
        data/${train_set} ${dict_tgt} > ${feat_tr_dir}/data.${src_case}_${tgt_case}.json
    local/data2json.sh --text data/${train_set_prefix}.fr.gtranslate/text.${tgt_case} --nlsyms ${nlsyms} \
        data/${train_set_prefix}.fr.gtranslate ${dict_tgt} > ${feat_tr_dir}/data_gtranslate.${src_case}_${tgt_case}.json
    local/data2json.sh --text data/${train_dev}/text.${tgt_case} --nlsyms ${nlsyms} \
        data/${train_dev} ${dict_tgt} > ${feat_dt_dir}/data.${src_case}_${tgt_case}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/data2json.sh --text data/${rtask}/text.${tgt_case} --nlsyms ${nlsyms} \
            data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data.${src_case}_${tgt_case}.json
    done

    # update json (add source references)
    local/update_json.sh --text data/"$(echo ${train_set} | cut -f -1 -d ".")".en/text.${src_case} --nlsyms ${nlsyms} \
        ${feat_tr_dir}/data.${src_case}_${tgt_case}.json data/"$(echo ${train_set} | cut -f -1 -d ".")".en ${dict_src}
    local/update_json.sh --text data/"$(echo ${train_set} | cut -f -1 -d ".")".en/text.${src_case} --nlsyms ${nlsyms} \
        ${feat_tr_dir}/data_gtranslate.${src_case}_${tgt_case}.json data/"$(echo ${train_set} | cut -f -1 -d ".")".en ${dict_src}
    local/update_json.sh --text data/"$(echo ${train_dev} | cut -f -1 -d ".")".en/text.${src_case} --nlsyms ${nlsyms} \
        ${feat_dt_dir}/data.${src_case}_${tgt_case}.json data/"$(echo ${train_dev} | cut -f -1 -d ".")".en ${dict_src}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        local/update_json.sh --text data/"$(echo ${rtask} | cut -f -1 -d ".")".en/text.${src_case} --nlsyms ${nlsyms} \
            ${feat_recog_dir}/data.${src_case}_${tgt_case}.json data/"$(echo ${rtask} | cut -f -1 -d ".")".en ${dict_src}
    done

    # concatenate Fr and Fr (Google translation) jsons
    local/concat_json_multiref.py \
        ${feat_tr_dir}/data.${src_case}_${tgt_case}.json \
        ${feat_tr_dir}/data_gtranslate.${src_case}_${tgt_case}.json > ${feat_tr_dir}/data_2ref.${src_case}_${tgt_case}.json
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${src_case}_${tgt_case}_${backend}_$(basename ${train_config%.*})
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
        --dict-src ${dict_src} \
        --dict-tgt ${dict_tgt} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_2ref.${src_case}_${tgt_case}.json \
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
            mt_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_bleu.sh --case ${tgt_case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} fr ${dict_tgt} ${dict_src}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

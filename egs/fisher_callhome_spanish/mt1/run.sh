#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from 0 if you need to start from data preparation
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

# data
sfisher_speech=/export/corpora/LDC/LDC2010S01
sfisher_transcripts=/export/corpora/LDC/LDC2010T04
split=local/splits/split_fisher

callhome_speech=/export/corpora/LDC/LDC96S35
callhome_transcripts=/export/corpora/LDC/LDC96T17
split_callhome=local/splits/split_callhome

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

train_set=train.en
train_set_prefix=train
train_dev=dev.en
recog_set="fisher_dev.en fisher_dev2.en fisher_test.en callhome_devtest.en callhome_evltest.en"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/fsp_data_prep.sh ${sfisher_speech} ${sfisher_transcripts}
    local/callhome_data_prep.sh ${callhome_speech} ${callhome_transcripts}

    # split data
    local/create_splits.sh ${split}
    local/callhome_create_splits.sh ${split_callhome}

    # concatenate multiple utterances
    local/normalize_trans.sh ${sfisher_transcripts} ${callhome_transcripts}
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    cp -rf data/fisher_train data/train

    # Divide into source and target languages
    for x in ${train_set_prefix} fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        local/divide_lang.sh ${x}
    done

    cp -rf data/fisher_dev.es data/dev.es
    cp -rf data/fisher_dev.en data/dev.en
    # NOTE: do not use callhome_train for the training set

    for x in ${train_set_prefix} dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in es en; do
            remove_longshortdata.sh --no_feat true --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f -1 -d " " data/${x}.es.tmp/text > data/${x}.en.tmp/reclist1
        cut -f -1 -d " " data/${x}.en.tmp/text > data/${x}.en.tmp/reclist2
        comm -12 data/${x}.en.tmp/reclist1 data/${x}.en.tmp/reclist2 > data/${x}.en.tmp/reclist

        for lang in es en; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.en.tmp/reclist data/${x}.${lang}
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
    local/data2json.sh --text data/${train_dev}/text.${tgt_case} --nlsyms ${nlsyms} \
        data/${train_dev} ${dict_tgt} > ${feat_dt_dir}/data.${src_case}_${tgt_case}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/data2json.sh --text data/${rtask}/text.${tgt_case} --nlsyms ${nlsyms} \
            data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data.${src_case}_${tgt_case}.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev} ${recog_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f -1 -d ".").es
        local/update_json.sh --text ${data_dir}/text.${src_case} --nlsyms ${nlsyms} \
            ${feat_dir}/data.${src_case}_${tgt_case}.json ${data_dir} ${dict_src}
    done

    # Fisher has 4 references per utterance
    for rtask in fisher_dev.en fisher_dev2.en fisher_test.en; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        for no in 1 2 3; do
          local/data2json.sh --text data/${rtask}/text.${tgt_case}.${no} --nlsyms ${nlsyms} \
              data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data_${no}.${src_case}_${tgt_case}.json
        done
    done
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
            mt_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        # Fisher has 4 references per utterance
        if [ ${rtask} = "fisher_dev.en" ] || [ ${rtask} = "fisher_dev2.en" ] || [ ${rtask} = "fisher_test.en" ]; then
            for no in 1 2 3; do
              cp ${feat_recog_dir}/data_${no}.${src_case}_${tgt_case}.json ${expdir}/${decode_dir}/data_ref${no}.json
            done
        fi

        local/score_bleu.sh --case ${tgt_case} --set ${rtask} --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict_tgt} ${dict_src}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

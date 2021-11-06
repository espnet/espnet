#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=8            # number of parallel jobs for decoding
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
n_average=5                  # the number of MT models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best MT models will be averaged.
                             # if false, the last `n_average` MT models will be averaged.
metric=bleu                  # loss/acc/bleu

# preprocessing related
src_case=tc
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default

# mtedx_datadir=download # original data directory to be stored
mtedx_datadir=/n/work3/inaguma/mTEDx # original data directory to be stored

# language related
src_lang=es
tgt_lang=en
# English (en)
# French (fr)
# German (de)
# Spanish (es)
# Italian (it)
# Russian (ru)
# Greek (el)
# Portuguese (pt)
# Arabic (ar)

# if true, reverse source and target languages: **->English
reverse_direction=false

# use the same dict as in the ST task
use_st_dict=true

# bpemode (unigram or bpe)
nbpe=1000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train.${src_lang}-${tgt_lang}.${tgt_lang}
train_dev=dev.${src_lang}-${tgt_lang}.${tgt_lang}
trans_set="valid.${src_lang}-${tgt_lang}.${tgt_lang} test.${src_lang}-${tgt_lang}.${tgt_lang}"

# verify language directions
is_exist=false
if [[ ${src_lang} == es ]]; then
    tgt=en_fr_it_pt_es
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
elif [[ ${src_lang} == fr ]]; then
    tgt=en_es_pt_fr
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
elif [[ ${src_lang} == it ]]; then
    tgt=en_es_it
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
elif [[ ${src_lang} == pt ]]; then
    tgt=en_es_pt
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
elif [[ ${src_lang} == ru ]]; then
    tgt=en_ru
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
elif [[ ${src_lang} == el ]]; then
    tgt=en_el
    for lang in $(echo ${tgt} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
fi
if [[ ${is_exist} == false ]]; then
    echo "No language direction: ${src_lang} to ${tgt_lang}" && exit 1;
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases

    local/data_prep.sh ${mtedx_datadir} ${src_lang} ${tgt_lang}
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Divide into source and target languages
    for x in train.${src_lang}-${tgt_lang} valid.${src_lang}-${tgt_lang} test.${src_lang}-${tgt_lang}; do
        divide_lang.sh ${x} "${src_lang} ${tgt_lang}"
    done
    for lang in ${src_lang} ${tgt_lang}; do
        cp -rf data/valid.${src_lang}-${tgt_lang}.${lang} data/dev.${src_lang}-${tgt_lang}.${lang}
    done

    # remove long and short utterances
    for x in train.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang}; do
        clean_corpus.sh --no_feat true --maxchars 400 --utt_extra_files "text.tc text.lc text.lc.rm" data/${x} "${src_lang} ${tgt_lang}"
    done
fi

if [ ${use_st_dict} = true ]; then
    if [ ${reverse_direction} = true ]; then
        dict=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_units_${src_case}.txt
        nlsyms=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_non_lang_syms_${src_case}.txt
        bpemodel=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${src_case}
    else
        dict=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_units_${tgt_case}.txt
        nlsyms=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_non_lang_syms_${tgt_case}.txt
        bpemodel=../st1/data/lang_1spm/train.${src_lang}-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${tgt_case}
    fi
else
    dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
    nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
    bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
fi
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    if [ ${use_st_dict} = false ]; then
        echo "make a non-linguistic symbol list for all languages"
        cut -f 2- -d' ' data/train.${src_lang}-${tgt_lang}.*/text.${tgt_case} | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
        cat ${nlsyms}

        echo "make a joint source and target dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        offset=$(wc -l < ${dict})
        cut -f 2- -d' ' data/train.${src_lang}-${tgt_lang}.*/text.${tgt_case} | grep -v -e '^\s*$' > data/lang_1spm/input_${src_lang}_${tgt_lang}_${src_case}_${tgt_case}.txt
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input_${src_lang}_${tgt_lang}_${src_case}_${tgt_case}.txt \
            --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input_${src_lang}_${tgt_lang}_${src_case}_${tgt_case}.txt \
            | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        wc -l ${dict}
    fi

    echo "make json files"
    if [ ${reverse_direction} = true ]; then
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}; mkdir -p ${feat_dir}
            set=$(echo ${x} | cut -f 1 -d ".")
            data2json.sh --nj 16 --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "${src_lang}" \
                data/${set} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${tgt_lang}
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
        done
    else
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}; mkdir -p ${feat_dir}
            data2json.sh --nj 16 --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "${tgt_lang}" \
                data/${x} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
        done
    fi
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${src_case}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
else
    expname=${train_set}_${src_case}_${tgt_case}_${backend}_${tag}
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
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average MT models
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

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_dir=${dumpdir}/${x}

        # reset log for RTF calculation
        if [ -f ${expdir}/${decode_dir}/log/decode.1.log ]; then
            rm ${expdir}/${decode_dir}/log/decode.*.log
        fi

        # split data
        splitjson.py --parts ${nj} ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
            --remove_nonverbal ${remove_nonverbal} \
            ${expdir}/${decode_dir} "${tgt_lang}" ${dict}

        calculate_rtf.py --log-dir ${expdir}/${decode_dir}/log
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

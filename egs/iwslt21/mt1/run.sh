#!/bin/bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=5
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

# if true, reverse source and target languages: **->English
reverse_direction=false

# use the same dict as in the ST task
use_st_dict=false

# bpemode (unigram or bpe)
nbpe=32000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

# data size related
datasize=10m  # 5m/10m/20m

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# data directories
mustc_dir=../../must_c
mustc_v2_dir=../../must_c_v2
stted_dir=../../iwslt18

if [ ${reverse_direction} = true ]; then
    train_set=train_${datasize}.en
    train_dev=dev.en
    trans_subset="et_mustc_dev_org.en et_mustc_tst-COMMON.en et_mustc_tst-HE.en"
    trans_set="et_mustc_dev_org.en et_mustc_tst-COMMON.en et_mustc_tst-HE.en \
               et_mustcv2_dev_org.en et_mustcv2_tst-COMMON.en et_mustcv2_tst-HE.en \
               et_stted_dev2010.en et_stted_tst2010.en et_stted_tst2013.en et_stted_tst2014.en et_stted_tst2015.en \
               et_stted_tst2018.en et_stted_tst2019.en"
else
    train_set=train_${datasize}.de
    train_dev=dev.de
    trans_subset="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de"
    trans_set="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de \
               et_mustcv2_dev_org.de et_mustcv2_tst-COMMON.de et_mustcv2_tst-HE.de \
               et_stted_dev2010.de et_stted_tst2010.de et_stted_tst2013.de et_stted_tst2014.de et_stted_tst2015.de \
               et_stted_tst2018.de et_stted_tst2019.de"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"

    # WMT20
    local/data_prep_wmt20.sh --max_length 250 --length_ratio 1.5 ${datasize}

    # Must-C
    if [ ! -d "${mustc_dir}/mt1/data/train.en-de.en" ]; then
        echo "run ${mustc_dir}/mt1/run.sh first"
        exit 1
    fi
    data_code=mustc
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/mt1/data/train.en-de.${lang}      data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/mt1/data/dev.en-de.${lang}        data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/mt1/data/dev_org.en-de.${lang}    data/et_${data_code}_dev_org.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/mt1/data/tst-COMMON.en-de.${lang} data/et_${data_code}_tst-COMMON.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/mt1/data/tst-HE.en-de.${lang}     data/et_${data_code}_tst-HE.${lang}
    done

    # Must-C v2
    if [ ! -d "${mustc_v2_dir}/mt1/data/train.en-de.en" ]; then
        echo "run ${mustc_v2_dir}/mt1/run.sh first"
        exit 1
    fi
    data_code=mustcv2
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/mt1/data/train.en-de.${lang}      data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/mt1/data/dev.en-de.${lang}        data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/mt1/data/dev_org.en-de.${lang}    data/et_${data_code}_dev_org.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/mt1/data/tst-COMMON.en-de.${lang} data/et_${data_code}_tst-COMMON.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/mt1/data/tst-HE.en-de.${lang}     data/et_${data_code}_tst-HE.${lang}
    done

    # ST-TED
    if [ ! -d "${stted_dir}/mt1/data/train_nodevtest.en" ]; then
        echo "run ${stted_dir}/mt1/run.sh first"
        exit 1
    fi
    data_code=stted
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/mt1/data/train_nodevtest.${lang} data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/mt1/data/train_dev.${lang}       data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/mt1/data/dev.${lang}             data/et_${data_code}_dev.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/mt1/data/test.${lang}            data/et_${data_code}_test.${lang}
    done
    # En-De only
    for x in dev2010 tst2010 tst2010 tst2013 tst2014 tst2015 tst2018 tst2019; do
        cp -rf ${stted_dir}/mt1/data/${x}.en data/et_${data_code}_${x}.en
        cp -rf ${stted_dir}/mt1/data/${x}.de data/et_${data_code}_${x}.de
    done
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    rm data/tr_*/segments data/dt_*/segments data/et_*/segments
    rm data/tr_*//wav.scp data/dt_*//wav.scp data/et_*//wav.scp

    # Divide into source and target languages
    divide_lang.sh tr_wmt20_subset${datasize} "en de"

    for lang in en de; do
        utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/train_${datasize}.${lang} data/tr_wmt20_subset${datasize}.${lang} data/tr_mustc.${lang} data/tr_mustcv2.${lang} data/tr_stted.${lang}
        cp -rf data/dt_mustc.${lang} data/dev.${lang}
    done

    echo "Remove offlimit"
    for lang in en de; do
        cp -rf data/train_${datasize}.${lang} data/train_${datasize}.${lang}.tmp
        cp data/train_${datasize}.${lang}/utt2spk data/train_${datasize}.${lang}/utt2spk.org
        local/filter_offlimit.py --offlimit_list local/offlimit_list --utt2spk data/train_${datasize}.${lang}/utt2spk.org > data/train_${datasize}.${lang}/utt2spk
        utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/train_${datasize}.${lang}
        rm -rf data/train_${datasize}.${lang}.tmp
        # NOTE: 5 speakers are expected to be removed
    done
fi

if [ ${use_st_dict} = true ]; then
    if [ ${reverse_direction} = true ]; then
        dict=../st1/data/lang_1spm/train_sp.en-de.de_${bpemode}${nbpe}_units_${src_case}.txt
        nlsyms=../st1/data/lang_1spm/train_sp.en-de.de_non_lang_syms_${src_case}.txt
        bpemodel=../st1/data/lang_1spm/train_sp.en-de.de_${bpemode}${nbpe}_${src_case}
    else
        dict=../st1/data/lang_1spm/train_sp.en-de.de_${bpemode}${nbpe}_units_${tgt_case}.txt
        nlsyms=../st1/data/lang_1spm/train_sp.en-de.de_non_lang_syms_${tgt_case}.txt
        bpemodel=../st1/data/lang_1spm/train_sp.en-de.de_${bpemode}${nbpe}_${tgt_case}
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
        cut -f 2- -d' ' data/train_${datasize}.*/text.${tgt_case} | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
        cat ${nlsyms}

        echo "make a joint source and target dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        offset=$(wc -l < ${dict})
        cut -f 2- -d' ' data/train_${datasize}.*/text.${tgt_case} | grep -v -e '^\s*$' > data/lang_1spm/input_${datasize}_${src_case}_${tgt_case}.txt
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input_${datasize}_${src_case}_${tgt_case}.txt \
            --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=0.9995
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input_${datasize}_${src_case}_${tgt_case}.txt \
            | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        wc -l ${dict}
    fi

    echo "make json files"
    if [ ${reverse_direction} = true ]; then
        data2json.sh --nj 16 --text data/train_${datasize}.en/text.${tgt_case} --bpecode ${bpemodel}.model --lang "en" \
            data/train_${datasize}.en ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        for x in ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}; mkdir -p ${feat_dir}
            set=$(echo ${x} | cut -f 1 -d ".")
            data2json.sh --text data/${set}.en/text.${tgt_case} --bpecode ${bpemodel}.model --lang "en" \
                data/${set}.en ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json
        done

        # update json (add source references)
        update_json.sh --text data/train_${datasize}.de/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json data/train_${datasize}.de ${dict}
        for x in ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").de
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json ${data_dir} ${dict}
        done
    else
        data2json.sh --nj 16 --text data/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "de" \
            data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        for x in ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}; mkdir -p ${feat_dir}
            data2json.sh --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "de" \
                data/${x} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json
        done

        # update json (add source references)
        update_json.sh --text data/train_${datasize}.en/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json data/train_${datasize}.en ${dict}
        for x in ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").en
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json ${data_dir} ${dict}
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
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json \
        --n-iter-processes 2
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
    for x in ${trans_subset}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_dir=${dumpdir}/${x}

        # split data
        splitjson.py --parts ${nj} ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}_${datasize}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        if [ ${reverse_direction} = true ]; then
            score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "en" ${dict}
        else
            score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "de" ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

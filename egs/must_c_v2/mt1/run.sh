#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=5
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=4            # number of parallel jobs for decoding
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

# cascaded-ST related
asr_model_dir=
decode_config_asr=
dict_asr=

# preprocessing related
src_case=lc.rm
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
must_c=/n/rd8/MUSTC_v2.0

# target language related
tgt_lang=de
# you can choose from de, es, fr, it, nl, pt, ro, ru

# if true, reverse source and target languages: **->English
reverse_direction=false

# use the same dict as in the ST task
use_st_dict=true

# bpemode (unigram or bpe)
nbpe=8000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${reverse_direction} = true ]; then
    train_set=train.${tgt_lang}-en.en
    train_dev=dev.${tgt_lang}-en.en
    trans_set="dev_org.${tgt_lang}-en.en tst-COMMON.${tgt_lang}-en.en tst-HE.${tgt_lang}-en.en"
else
    train_set=train.en-${tgt_lang}.${tgt_lang}
    train_dev=dev.en-${tgt_lang}.${tgt_lang}
    trans_set="dev_org.en-${tgt_lang}.${tgt_lang} tst-COMMON.en-${tgt_lang}.${tgt_lang} tst-HE.en-${tgt_lang}.${tgt_lang}"
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${must_c} ${tgt_lang} "v2"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/data_prep.sh ${must_c} ${tgt_lang} "v2"
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Divide into source and target languages
    for x in train.en-${tgt_lang} dev.en-${tgt_lang} tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}; do
        divide_lang.sh ${x} "en ${tgt_lang}"
    done
    for lang in ${tgt_lang} en; do
        cp -rf data/dev.en-${tgt_lang}.${lang} data/dev_org.en-${tgt_lang}.${lang}
    done

    # remove long and short utterances
    for x in train.en-${tgt_lang} dev.en-${tgt_lang}; do
        clean_corpus.sh --no_feat true --maxchars 400 --utt_extra_files "text.tc text.lc text.lc.rm" data/${x} "en ${tgt_lang}"
    done
fi

if [ ${use_st_dict} = true ]; then
    if [ ${reverse_direction} = true ]; then
        dict=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_units_${src_case}.txt
        nlsyms=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_non_lang_syms_${src_case}.txt
        bpemodel=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${src_case}
    else
        dict=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_units_${tgt_case}.txt
        nlsyms=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_non_lang_syms_${tgt_case}.txt
        bpemodel=../st1/data/lang_1spm/train_sp.en-${tgt_lang}.${tgt_lang}_${bpemode}${nbpe}_${tgt_case}
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
        cut -f 2- -d' ' data/train.en-${tgt_lang}.*/text.${tgt_case} | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
        cat ${nlsyms}

        echo "make a joint source and target dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        offset=$(wc -l < ${dict})
        cut -f 2- -d' ' data/train.en-${tgt_lang}.*/text.${tgt_case} | grep -v -e '^\s*$' > data/lang_1spm/input_${tgt_lang}_${src_case}_${tgt_case}.txt
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input_${tgt_lang}_${src_case}_${tgt_case}.txt \
            --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input_${tgt_lang}_${src_case}_${tgt_case}.txt \
            | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        wc -l ${dict}
    fi

    echo "make json files"
    if [ ${reverse_direction} = true ]; then
        data2json.sh --nj 16 --text data/train.en-${tgt_lang}.en/text.${tgt_case} --bpecode ${bpemodel}.model --lang "en" \
            data/train.en-${tgt_lang}.en ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        for x in ${train_dev} ${trans_set}; do
            feat_trans_dir=${dumpdir}/${x}; mkdir -p ${feat_trans_dir}
            set=$(echo ${x} | cut -f 1 -d ".")
            data2json.sh --text data/${set}.en-${tgt_lang}.en/text.${tgt_case} --bpecode ${bpemodel}.model --lang "en" \
                data/${set}.en-${tgt_lang}.en ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${tgt_lang}.${tgt_lang}
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
        done
    else
        data2json.sh --nj 16 --text data/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "${tgt_lang}" \
            data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        for x in ${train_dev} ${trans_set}; do
            feat_trans_dir=${dumpdir}/${x}; mkdir -p ${feat_trans_dir}
            data2json.sh --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "${tgt_lang}" \
                data/${x} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        done

        # update json (add source references)
        for x in ${train_set} ${train_dev} ${trans_set}; do
            feat_dir=${dumpdir}/${x}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${tgt_lang}.en
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
        feat_trans_dir=${dumpdir}/${x}

        # reset log for RTF calculation
        if [ -f ${expdir}/${decode_dir}/log/decode.1.log ]; then
            rm ${expdir}/${decode_dir}/log/decode.*.log
        fi

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        if [ ${reverse_direction} = true ]; then
            score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "en" ${dict}
        else
            score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "${tgt_lang}" ${dict}
        fi

        calculate_rtf.py --log-dir ${expdir}/${decode_dir}/log
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && [ -n "${asr_model_dir}" ] && [ -n "${decode_config_asr}" ] && [ -n "${dict_asr}" ] && [ ${reverse_direction} = false ]; then
    echo "stage 6: Cascaded-ST decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average MT models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
        else
            trans_model=model.last${n_average}.avg.best
        fi
    fi

    for x in ${trans_set}; do
        feat_trans_dir=${expdir}/$(echo ${asr_model_dir} | rev | cut -f 1 -d "/" | rev)/${x}; mkdir -p ${feat_trans_dir}
        rtask=$(echo ${x} | cut -f -2 -d ".").en
        data_dir=data/${rtask}

        # ASR outputs
        asr_decode_dir=decode_${rtask}_$(basename ${decode_config_asr%.*})
        json2text.py ${asr_model_dir}/${asr_decode_dir}/data.json ${dict_asr} ${data_dir}/text_asr_ref.${src_case} ${data_dir}/text_asr_hyp.${src_case}
        paste -d " " <(cut -d " " -f 1 ${data_dir}/text_asr_hyp.${src_case}) <(cut -d " " -f 2- ${data_dir}/text_asr_hyp.${src_case} | spm_decode --model=${bpemodel}.model --input_format=piece | sed -e "s/▁/ /g" | awk '{if(NF>0) {print $0;} else {print "emptyutterance";}}') \
            > ${data_dir}/text_asr_hyp.wrd.${src_case}

        data2json.sh --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "${tgt_lang}" \
            data/${x} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        update_json.sh --text ${data_dir}/text_asr_hyp.wrd.${src_case} --bpecode ${bpemodel}.model \
            ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
    done

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})_pipeline
        feat_trans_dir=${expdir}/$(echo ${asr_model_dir} | rev | cut -f 1 -d "/" | rev)/${x}

        # reset log for RTF calculation
        if [ -f ${expdir}/${decode_dir}/log/decode.1.log ]; then
            rm ${expdir}/${decode_dir}/log/decode.*.log
        fi

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
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

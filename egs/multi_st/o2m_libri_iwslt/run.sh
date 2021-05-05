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

# pre-training related
asr_model=
mt_model=

# preprocessing related
case=lc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
st_ted=/export/b08/inaguma/IWSLT
# st_ted=/n/rd11/corpora_8/iwslt18

# data directories
librispeech=../../libri_trans
iwslt18=../../iwslt18

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

train_set=train
train_dev=train_dev
recog_set_libri="et_librispeech_dev.fr et_librispeech_test.fr"
recog_set_iwslt="et_iwslt18_dev.de et_iwslt18_test.de\
 et_iwslt18_dev2010.de et_iwslt18_tst2010.de et_iwslt18_tst2013.de et_iwslt18_tst2014.de et_iwslt18_tst2015.de"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Librispeech
    if [ ! -d "${librispeech}/st1/data" ]; then
        echo "run ${librispeech}/st1/run.sh first"
        exit 1
    fi
    lang_code=librispeech
    for lang in en fr; do
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${librispeech}/st1/data/train_sp.${lang}  data/tr_${lang_code}.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${librispeech}/st1/data/train_dev.${lang} data/dt_${lang_code}.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${librispeech}/st1/data/dev.${lang}       data/et_${lang_code}_dev.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${librispeech}/st1/data/test.${lang}      data/et_${lang_code}_test.${lang}
    done
    local/copy_data_dir.sh --utt-suffix -${lang_code} ${librispeech}/st1/data/train_sp.fr.gtranslate data/tr_${lang_code}.fr.gtranslate

    # append language ID
    for lang in en fr; do
        for x in data/*"${lang_code}"*."${lang}"; do
            for c in tc lc lc.rm; do
                cp -rf ${x} ${x}.tmp
                awk -v lang="<2${lang}>" '{$2=lang""$2; print}' ${x}.tmp/text.${c} > ${x}/text.${c}
                rm -rf ${x}.tmp
            done
        done
    done
    cp -rf data/tr_${lang_code}.fr.gtranslate data/tr_${lang_code}.fr.gtranslate.tmp
    for c in tc lc lc.rm; do
        awk -v lang="<2fr>" '{$2=lang""$2; print}' data/tr_${lang_code}.fr.gtranslate.tmp/text.${c} > data/tr_${lang_code}.fr.gtranslate/text.${c}
    done
    rm -rf data/tr_${lang_code}.fr.gtranslate.tmp

    # IWSLT18
    lang_code=iwslt18
    if [ ! -d "${iwslt18}/st1/data" ]; then
        echo "run ${iwslt18}/st1/run.sh first"
        exit 1
    fi
    for lang in en de; do
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${iwslt18}/st1/data/train_nodevtest_sp.${lang} data/tr_${lang_code}.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${iwslt18}/st1/data/train_dev.${lang}          data/dt_${lang_code}.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${iwslt18}/st1/data/dev.${lang}                data/et_${lang_code}_dev.${lang}
        local/copy_data_dir.sh --utt-suffix -${lang_code} ${iwslt18}/st1/data/test.${lang}               data/et_${lang_code}_test.${lang}
        for rtask in dev2010 tst2010 tst2013 tst2014 tst2015; do
            local/copy_data_dir.sh ${iwslt18}/st1/data/${rtask}.${lang} data/et_${lang_code}_${rtask}.${lang}
            # append language ID
            awk -v lang="<2${lang}>" '{$2=lang""$2; print}' ${iwslt18}/st1/data/${rtask}.${lang}/text_noseg.tc > data/et_${lang_code}_${rtask}.${lang}/text_noseg.tc
            awk -v lang="<2${lang}>" '{$2=lang""$2; print}' ${iwslt18}/st1/data/${rtask}.${lang}/text_noseg.lc > data/et_${lang_code}_${rtask}.${lang}/text_noseg.lc
            awk -v lang="<2${lang}>" '{$2=lang""$2; print}' ${iwslt18}/st1/data/${rtask}.${lang}/text_noseg.lc.rm > data/et_${lang_code}_${rtask}.${lang}/text_noseg.lc.rm
        done
    done

    # append language ID
    for lang in en de; do
        for x in data/*"${lang_code}"*."${lang}"; do
            for c in tc lc lc.rm; do
                cp -rf ${x} ${x}.tmp
                awk -v lang="<2${lang}>" '{$2=lang""$2; print}' ${x}.tmp/text.${c} > ${x}/text.${c}
                rm -rf ${x}.tmp
            done
        done
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_set} data/tr_librispeech.fr data/tr_iwslt18.de
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_dev} data/dt_librispeech.fr data/dt_iwslt18.de

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/multi_st/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/multi_st/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for rtask in ${recog_set_libri} ${recog_set_iwslt}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units_${case}.txt
nlsyms=data/lang_1char/non_lang_syms_${case}.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/tr_librispeech*/text.${case} data/tr_iwslt18*/text.${case} | cut -f 2- -d " " | grep -o -P '&[^;]*;|<[^>]*>' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    grep sp1.0 data/tr_librispeech*/text.${case} data/tr_iwslt18*/text.${case} | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    ### train
    # libri
    grep librispeech ${feat_tr_dir}/feats.scp > ${feat_tr_dir}/feats_libri.scp
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats_libri.scp --text data/tr_librispeech.fr/text.${case} --nlsyms ${nlsyms} \
        data/tr_librispeech.fr ${dict} > ${feat_tr_dir}/data_libri.${case}.json
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats_libri.scp --text data/tr_librispeech.fr.gtranslate/text.${case} --nlsyms ${nlsyms} \
        data/tr_librispeech.fr.gtranslate ${dict} > ${feat_tr_dir}/data_libri_gtranslate.${case}.json
    # add source references
    local/update_json.sh --nj 16 --text data/tr_librispeech.en/text.${case} --nlsyms ${nlsyms} \
        ${feat_tr_dir}/data_libri.${case}.json data/tr_librispeech.en ${dict}
    local/update_json.sh --nj 16 --text data/tr_librispeech.en/text.${case} --nlsyms ${nlsyms} \
        ${feat_tr_dir}/data_libri_gtranslate.${case}.json data/tr_librispeech.en ${dict}

    # iwslt
    grep iwslt18 ${feat_tr_dir}/feats.scp > ${feat_tr_dir}/feats_iwslt.scp
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats_iwslt.scp --text data/tr_iwslt18.de/text.${case} --nlsyms ${nlsyms} \
        data/tr_iwslt18.de ${dict} > ${feat_tr_dir}/data_iwslt.${case}.json
    # add source references
    local/update_json.sh --nj 16 --text data/tr_iwslt18.en/text.${case} --nlsyms ${nlsyms} \
        ${feat_tr_dir}/data_iwslt.${case}.json data/tr_iwslt18.en ${dict}

    # concatenate libri and iwslt
    local/concat_json_multiref.py \
        ${feat_tr_dir}/data_libri.${case}.json \
        ${feat_tr_dir}/data_libri_gtranslate.${case}.json \
        ${feat_tr_dir}/data_iwslt.${case}.json > ${feat_tr_dir}/data.${case}.json

    ### dev
    # libri
    grep librispeech ${feat_dt_dir}/feats.scp > ${feat_dt_dir}/feats_libri.scp
    local/data2json.sh --feat ${feat_dt_dir}/feats_libri.scp --text data/dt_librispeech.fr/text.${case} --nlsyms ${nlsyms} \
        data/dt_librispeech.fr ${dict} > ${feat_dt_dir}/data_libri.${case}.json
    # add source references
    local/update_json.sh --text data/dt_librispeech.en/text.${case} --nlsyms ${nlsyms} \
        ${feat_dt_dir}/data_libri.${case}.json data/dt_librispeech.en ${dict}

    # iwslt
    grep iwslt18 ${feat_dt_dir}/feats.scp > ${feat_dt_dir}/feats_iwslt.scp
    local/data2json.sh --feat ${feat_dt_dir}/feats_iwslt.scp --text data/dt_iwslt18.de/text.${case} --nlsyms ${nlsyms} \
        data/dt_iwslt18.de ${dict} > ${feat_dt_dir}/data_iwslt.${case}.json
    # add source references
    local/update_json.sh --text data/dt_iwslt18.en/text.${case} --nlsyms ${nlsyms} \
        ${feat_dt_dir}/data_iwslt.${case}.json data/dt_iwslt18.en ${dict}

    # concatenate libri and iwslt
    local/concat_json_multiref.py \
        ${feat_dt_dir}/data_libri.${case}.json \
        ${feat_dt_dir}/data_iwslt.${case}.json > ${feat_dt_dir}/data.${case}.json

    for rtask in ${recog_set_libri} ${recog_set_iwslt}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        if echo ${rtask} | grep 'dev2010' || echo ${rtask} | grep 'tst'; then
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --no_text true \
                data/${rtask} ${dict} > ${feat_recog_dir}/data.${case}.json
        else
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --text data/${rtask}/text.${case} --nlsyms ${nlsyms} \
                data/${rtask} ${dict} > ${feat_recog_dir}/data.${case}.json
        fi
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${case}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
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
        --train-json ${feat_tr_dir}/data.${case}.json \
        --valid-json ${feat_dt_dir}/data.${case}.json \
        --asr-model ${asr_model} \
        --mt-model ${mt_model} \
        --replace-sos true
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16

    pids=() # initialize pids
    for rtask in ${recog_set_libri} ${recog_set_iwslt}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.${case}.json

        #### use CPU for decoding
        ngpu=0

        if echo ${rtask} | grep 'libri'; then
            tgt_lang="\<2fr\>"
        elif echo ${rtask} | grep 'iwslt'; then
            tgt_lang="\<2de\>"
        fi

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --tgt-lang ${tgt_lang}

        if echo ${rtask} | grep 'libri'; then
            score_bleu.sh --case ${case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} fr ${dict}
        elif echo ${rtask} | grep 'iwslt'; then
            if [ ${rtask} = "et_iwslt18_dev.de" ] || [ ${rtask} = "et_iwslt18_test.de" ]; then
                score_bleu.sh --case ${case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} de ${dict}
            else
                set=$(echo ${rtask} | cut -f 1 -d "." | cut -f 3 -d "_")
                local/score_bleu_iwslt_reseg.sh --case ${case} --nlsyms ${nlsyms} \
                    ${expdir}/${decode_dir} ${dict} ${st_ted} ${set}
            fi
        fi

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

#!/usr/bin/env bash

# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang_pairs="en2de"

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${COVOST2}" ]; then
    log "Fill the value of 'COVOST2' of db.sh"
    exit 1
fi
mkdir -p ${COVOST2}

if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi
mkdir -p ${COMMONVOICE}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Downloading"

    for lang_pair in "$(echo ${lang_pairs} | tr '_' ' ')"; do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        # base url for downloads.
        data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/${src_lang}.tar.gz

        # Download CommonVoice
        mkdir -p ${COMMONVOICE}/${src_lang}
        local/download_and_untar_commonvoice.sh ${COMMONVOICE}/${src_lang} ${data_url} ${src_lang}.tar.gz

        # Download translation
        wget --no-check-certificate https://dl.fbaipublicfiles.com/covost/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz \
            -P ${COVOST2}
        tar -xzf ${COVOST2}/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz -C ${COVOST2}
        # wget --no-check-certificate https://dl.fbaipublicfiles.com/covost/covost2.zip \
        #       -P ${COVOST2}
        # unzip ${COVOST2}/covost2.zip -d ${COVOST2}
        # NOTE: some non-English target languages lack translation from English
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    # use underscore-separated names in data directories.
    for lang_pair in "$(echo ${lang_pairs} | tr '_' ' ')"; do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        local/data_prep_commonvoice.pl "${COMMONVOICE}/${src_lang}" validated data/validated.${src_lang}

        # text preprocessing (tokenization, case, punctuation marks etc.)
        local/data_prep_covost2.sh ${COVOST2} ${src_lang} ${tgt_lang} || exit 1;
        # NOTE: train/dev/test splits are different from original CommonVoice
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: New Splits"

    for lang_pair in $(echo ${lang_pairs} | tr '_' ' '); do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        local/create_splits.py ${src_lang} ${tgt_lang}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: New dataset creation"

    for lang_pair in $(echo ${lang_pairs} | tr '_' ' '); do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        python local/create_new_dataset.py ${src_lang} ${tgt_lang}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Clean file"

    for lang_pair in $(echo ${lang_pairs} | tr '_' ' '); do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        for set in train dev test; do
            dst_folder=data/${set}.${src_lang}-${tgt_lang}_new
            mv ${dst_folder}/wav.scp ${dst_folder}/wav_old.scp
            mv ${dst_folder}/wav_new.scp ${dst_folder}/wav.scp
            utils/utt2spk_to_spk2utt.pl < ${dst_folder}/utt2spk > ${dst_folder}/spk2utt
            mv data/${set}.${src_lang}-${tgt_lang} data/${set}.${src_lang}-${tgt_lang}_old
            mv data/${set}.${src_lang}-${tgt_lang}_new data/${set}.${src_lang}-${tgt_lang}
        done
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Create multilingual data directories"

    extra_files="text.lc.src text.lc.tgt "
    extra_files+="text.lc.rm.src text.lc.rm.tgt "
    extra_files+="text.tc.src text.tc.tgt "
    extra_files+="src_file.txt tgt_file.txt"

    for set in train dev test; do
        data_dirs=
        for lang_pair in $(echo ${lang_pairs} | tr '_' ' '); do
            src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
            tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")
            data_dir=data/${set}.${src_lang}-${tgt_lang}

            for file in ${data_dir}/text.*.${src_lang}; do
                cp ${file} $(echo ${file} | sed "s/.${src_lang}$/.src/g")
            done
            for file in ${data_dir}/text.*.${tgt_lang}; do
                cp ${file} $(echo ${file} | sed "s/.${tgt_lang}$/.tgt/g")
            done

            awk -v token="<${src_lang}>" '{$2=token; print}' ${data_dir}/utt2spk > ${data_dir}/src_file.txt
            awk -v token="<${tgt_lang}>" '{$2=token; print}' ${data_dir}/utt2spk > ${data_dir}/tgt_file.txt

            if [ "${set}" = "dev" ] || [ "${set}" = "test" ]; then
                utils/combine_data.sh --extra-files "${extra_files}" data/${set}.${lang_pair} ${data_dir}
            fi

            data_dirs+="${data_dir} "
        done
        if [ "${set}" = "train" ] || [ "${set}" = "dev" ]; then
            utils/combine_data.sh --extra-files "${extra_files}" data/${set}.${lang_pairs} ${data_dirs}
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

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
lang_pairs="es2en"

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ -z "${MULTILINGUAL_TEDX}" ]; then
    log "Fill the value of 'MULTILINGUAL_TEDX' of db.sh"
    exit 1
fi
mkdir -p ${MULTILINGUAL_TEDX}

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
        data_url="https://openslr.org/resources/100/mtedx_${src_lang}-${tgt_lang}.tgz"

        # Download translation dataset
        wget -c --progress=dot:giga ${data_url} -O "${MULTILINGUAL_TEDX}/mtedx_${src_lang}-${tgt_lang}.tgz"
        tar -xzf ${MULTILINGUAL_TEDX}/mtedx_${src_lang}-${tgt_lang}.tgz -C ${MULTILINGUAL_TEDX}/
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    # use underscore-separated names in data directories.
    for lang_pair in "$(echo ${lang_pairs} | tr '_' ' ')"; do
        src_lang=$(echo ${lang_pair} | cut -f 1 -d"2")
        tgt_lang=$(echo ${lang_pair} | cut -f 2 -d"2")

        local/data_prep_mtedx.sh ${MULTILINGUAL_TEDX}/ ${src_lang} ${tgt_lang}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Create multilingual data directories"

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

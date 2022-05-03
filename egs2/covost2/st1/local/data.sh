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
src_lang=es
tgt_lang=en

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
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    # use underscore-separated names in data directories.
    local/data_prep_commonvoice.pl "${COMMONVOICE}/${src_lang}" validated data/validated.${src_lang}

    # text preprocessing (tokenization, case, punctuation marks etc.)
    local/data_prep_covost2.sh ${COVOST2} ${src_lang} ${tgt_lang} || exit 1;
    # NOTE: train/dev/test splits are different from original CommonVoice
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: New Splits"

    local/create_splits.py ${src_lang} ${tgt_lang}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: New dataset creation"

    python local/create_new_dataset.py ${src_lang} ${tgt_lang}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Clean file"
    for set in train dev test; do
        dst_folder=data/${set}.${src_lang}-${tgt_lang}_new
        mv ${dst_folder}/wav.scp ${dst_folder}/wav_old.scp
        mv ${dst_folder}/wav_new.scp ${dst_folder}/wav.scp
        utils/utt2spk_to_spk2utt.pl < ${dst_folder}/utt2spk > ${dst_folder}/spk2utt
        mv data/${set}.${src_lang}-${tgt_lang} data/${set}.${src_lang}-${tgt_lang}_old
        mv data/${set}.${src_lang}-${tgt_lang}_new data/${set}.${src_lang}-${tgt_lang}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

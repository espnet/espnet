#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
version=c # c or t (please refer to cvss paper for details)

 . utils/parse_options.sh || exit 1;

# base url for download commonvoice
cv_data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/${src_lang}.tar.gz
cvss_data_url=https://storage.googleapis.com/cvss/cvss_${version}_v1.0/cvss_${version}_${src_lang}_en_v1.0.tar.gz


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${CVSS}
if [ -z "${CVSS}" ]; then
    log "Fill the value of 'CVSS' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${CVSS}"
    log "Prepare source data from commonvoice 4.0"
    mkdir -p ${CVSS}/commonvoice4/${src_lang}
    local/download_and_untar.sh ${CVSS}/commonvoice4/${src_lang} ${cv_data_url} ${src_lang}.tar.gz
    mkdir -p ${CVSS}/${src_lang}_en-${version}
    local/download_and_untar.sh ${CVSS}/${src_lang}_en-${version} ${cvss_data_url} cvss_${version}_${src_lang}_en_v1.0.tar.gz
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for commonvoice and cvss"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "train" "test" "dev"; do

        log "Prepare Commonvoice ${part}"
        if [ "${part}" = train ]; then
            local/cv_data_prep.pl \
                "${CVSS}/commonvoice4/${src_lang}" \
                validated data/"validated_${src_lang}" ${src_lang}
            mv data/"validated_${src_lang}" data/train_"${src_lang}"
        else
            local/cv_data_prep.pl \
                "${CVSS}/commonvoice4/${src_lang}" \
                ${part} data/"${part}_${src_lang}" ${src_lang}
        fi

        log "Prepare CVSS ${part}"
        python local/cvss_data_prep.py \
            --datadir "${CVSS}/${src_lang}_en-${version}" \
            --subset ${part} \
            --dest data/"${part}_${src_lang}" \
            --src_lang ${src_lang}

        ln -sf text.en data/"${part}_${src_lang}"/text
        ln -sf wav.scp.en data/"${part}_${src_lang}"/wav.scp

        utt_extra_files="wav.scp.${src_lang} wav.scp.en text.${src_lang} text.en"
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" data/${part}_${src_lang}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

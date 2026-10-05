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
# CommonVoice is not distributed through a public URL anymore. Set this to the
# (time-limited, signed) link obtained after accepting the terms on
# https://commonvoice.mozilla.org/en/datasets to download it automatically,
# or see the message printed by local/download_commonvoice.sh.
cv_data_url=

 . utils/parse_options.sh || exit 1;


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
    local/download_commonvoice.sh \
        "${CVSS}/commonvoice4/${src_lang}" "${src_lang}" cv-corpus-4-2019-12-10 "${cv_data_url}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for commonvoic"
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

        ln -sf text.es data/"${part}_${src_lang}"/text
        ln -sf wav.scp.es data/"${part}_${src_lang}"/wav.scp

        utils/fix_data_dir.sh data/${part}_${src_lang}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

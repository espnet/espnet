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
lang=te # te ta gu

. utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${MS_INDIC_IS18}" ]; then
    log "Fill the value of 'MS_INDIC_IS18' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [[ ! -d "${MS_INDIC_IS18}/${lang}-in-Train" ]]; then
        log "stage0: Download training data to ${MS_INDIC_IS18}. ${lang}-in-Train directory is missing"
        exit 1
    elif [[ ! -d "${MS_INDIC_IS18}/${lang}-in-Test" ]]; then
        log "stage0: Download test data to ${MS_INDIC_IS18}. ${lang}-in-Test directory is missing"
        exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing data for Microsoft Speech Corpus (Indian languages)"
    ### Task dependent. You have to make data the following preparation part by yourself.
    local/prepare_data.py ${MS_INDIC_IS18} ${lang}
fi


log "Successfully finished. [elapsed=${SECONDS}s]"

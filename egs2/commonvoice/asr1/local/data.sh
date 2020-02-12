#!/bin/bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang=cy # en de fr cy tt kab ca zh-TW it fa eu es ru

# base url for downloads.
data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/$lang.tar.gz

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir ${COMMONVOICE}
if [ ! -e "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=valid_train_${lang}
train_dev=valid_dev_${lang}
test_set=valid_test_${lang}

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Download data to ${COMMONVOICE}"
    mkdir -p ${COMMONVOICE}
    local/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases 
    for part in "validated"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl ${COMMONVOICE} ${part} data/"$(echo "${part}_${lang}" | tr - _)"
    done

    # Kaldi Version Split
    # ./utils/subset_data_dir_tr_cv.sh data/validated data/valid_train data/valid_test_dev
    # ./utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50 data/valid_test_dev data/valid_test data/valid_dev

    # ESPNet Version (same as voxforge)
    # consider duplicated sentences (does not consider speaker split)
    # filter out the same sentences (also same text) of test&dev set from validated set
    local/split_tr_dt_et.sh data/validated_${lang} data/${train_set} data/${train_dev} data/${test_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

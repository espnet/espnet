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
lang=en # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk 

 . utils/parse_options.sh || exit 1;

# base url for downloads.
# Deprecated url:https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/$lang.tar.gz
data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/${lang}.tar.gz

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${COMMONVOICE}
if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set=test_"$(echo "${lang}" | tr - _)"

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Download data to ${COMMONVOICE}"
    log "The default data of this recipe is from commonvoice 5.1, for newer version, you need to register at \
         https://commonvoice.mozilla.org/"
    local/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${COMMONVOICE}/cv-corpus-5.1-2020-06-22/${lang}" ${part} data/"$(echo "${part}_${lang}" | tr - _)"
    done

    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set}
    utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp
    utils/fix_data_dir.sh data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

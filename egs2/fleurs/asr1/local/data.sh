#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
nlsyms_txt=data/nlsyms.txt
SECONDS=0
lang=af_za # see https://huggingface.co/datasets/google/fleurs#dataset-structure for list of all langs
asr_data_dir=

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${FLEURS}
if [ -z "${FLEURS}" ]; then
    log "Fill the value of 'FLEURS' of db.sh"
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
    log "stage1: Download data to ${FLEURS}"
    mkdir -p "downloads/${lang}"
    python local/create_dataset.py --lang ${lang} --nlsyms_txt ${nlsyms_txt}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for fleurs"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "train" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${FLEURS}/${lang}" ${part} data/"$(echo "${part}_${lang}" | tr - _)"
    done

    # remove test&dev data from validated sentences
    #utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set}
    utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp
    utils/fix_data_dir.sh data/${train_set}
fi

if [ ${stage} -eq 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage3: Additional data processing - This should only be called after ASR stage 4"
    # create file of lids for self-conditioning
    python local/create_lids.py --data_dir ${asr_data_dir}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

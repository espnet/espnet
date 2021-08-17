#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

 log() {
     local fname=${BASH_SOURCE[1]##*/}
     echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
 }

 if [ ! -e "${MUCS_SUBTASK1}" ]; then
     log "Specify path for data in db.sh."
     log "Download data from Tamil, Telugu & Gujarati from https://navana-tech.github.io/MUCS2021/data.html."
     log "Place it inside data path."
     exit 1
 fi

 if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     log "stage1: Download data to ${MUCS_SUBTASK1}"
     mkdir -p ${MUCS_SUBTASK1}
     local/download_data.sh ${MUCS_SUBTASK1}
 fi

 if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     log "stage2: Preparing data for MUCS subtask1"
     ### Task dependent. You have to make data the following preparation part by yourself.
    mkdir -p data
    local/prepare_data.sh ${MUCS_SUBTASK1}
    local/check_audio_data_folder.sh ${MUCS_SUBTASK1}
    local/test_data_prep.sh ${MUCS_SUBTASK1} data/test
    local/train_data_prep.sh ${MUCS_SUBTASK1} data/train

 fi

 log "Successfully finished. [elapsed=${SECONDS}s]"


#

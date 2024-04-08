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
lang=te  # te, ta
 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ -z "${MICROSOFT_SPEECH_CORPUS}" ]; then
    log "Fill the value of 'MICROSOFT_SPEECH_CORPUS' of db.sh"
    exit 1
fi



# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage1: Download data to ${MICROSOFT_SPEECH_CORPUS}"
    log "Download data from the link:  https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e"
    log "checking if the right directory structure exists"

    if [ -d "${MICROSOFT_SPEECH_CORPUS}/${lang}-in-Train/Audios" ]
    then
        echo "Data directory exists."
    else
        echo "Error: Directory ${MICROSOFT_SPEECH_CORPUS}/${lang}-in-Train/Audios does not exists."
    fi
fi


mkdir -p data
mkdir -p data/dev_${lang}
mkdir -p data/test_${lang}
mkdir -p data/train_${lang}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for microsoft_speech_corpus"
    python local/process.py ${MICROSOFT_SPEECH_CORPUS} ${lang}
    ### Running python script for preparing data in Kaldi style from Microsoft speech corpus

fi

log "Successfully finished. [elapsed=${SECONDS}s]"

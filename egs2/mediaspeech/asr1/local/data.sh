#!/bin/bash

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=1
# inclusive, was 100
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

log "data preparation started"
mkdir -p ${MEDIASPEECH}
if [ -z "${MEDIASPEECH}" ]; then
    log "Fill the value of 'MEDIASPEECH' of db.sh"
    exit 1
fi

workspace=$PWD
lang=ES

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    cd ${MEDIASPEECH}
    wget https://www.openslr.org/resources/108/${lang}.tgz
    tar -xvf ${lang}.tgz
    rm -f ${lang}.tgz
    mv ${lang}/* .
    rm -rf ${lang}
    cd $workspace
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/data_prep.py -d ${MEDIASPEECH}
    utils/spk2utt_to_utt2spk.pl data/mediaspeech_train/spk2utt > data/mediaspeech_train/utt2spk
    utils/spk2utt_to_utt2spk.pl data/mediaspeech_dev/spk2utt > data/mediaspeech_dev/utt2spk
    utils/spk2utt_to_utt2spk.pl data/mediaspeech_test/spk2utt > data/mediaspeech_test/utt2spk
    utils/fix_data_dir.sh data/mediaspeech_train
    utils/fix_data_dir.sh data/mediaspeech_dev
    utils/fix_data_dir.sh data/mediaspeech_test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

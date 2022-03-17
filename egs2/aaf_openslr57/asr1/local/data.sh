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

DATA_DEST=data_asr
mkdir -p ${DATA_DEST}
mkdir -p ${DATA_DEST}/train
mkdir -p ${DATA_DEST}/test
mkdir -p ${DATA_DEST}/valid

DATA_KALDI=data
mkdir -p ${DATA_KALDI}
mkdir -p ${DATA_KALDI}/train
mkdir -p ${DATA_KALDI}/test
mkdir -p ${DATA_KALDI}/valid

workspace=$PWD

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    wget https://www.openslr.org/resources/57/African_Accented_French.tar.gz
    tar -xvf African_Accented_French.tar.gz
    rm -f African_Accented_French.tar.gz

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/data_prep_aaf.py --data-src African_Accented_French --data-dest ${DATA_DEST} --data-kaldi ${DATA_KALDI}
    rm -r African_Accented_French

    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
    utils/utt2spk_to_spk2utt.pl data/valid/utt2spk > data/valid/spk2utt

    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/valid
    utils/fix_data_dir.sh data/test

    utils/validate_data_dir.sh data/train --no-feats
    utils/validate_data_dir.sh data/valid --no-feats
    utils/validate_data_dir.sh data/test --no-feats
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
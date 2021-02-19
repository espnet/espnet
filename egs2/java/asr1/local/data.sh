#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
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

mkdir -p ${JAVA}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    wget https://www.openslr.org/resources/35/asr_javanese_0.zip
    mv asr_javanese_0.zip downloads
    unzip downloads/asr_javanese_0.zip
    rm -f downloads/asr_javanese_0.zip
    mv asr_javanese/* downloads
    rm -rf asr_javanese
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/java_data_prep.py
    utils/spk2utt_to_utt2spk.pl data/train/spk2utt > data/train/utt2spk
    utils/spk2utt_to_utt2spk.pl data/dev/spk2utt > data/dev/utt2spk
    utils/spk2utt_to_utt2spk.pl data/test/spk2utt > data/test/utt2spk
    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/dev
    utils/fix_data_dir.sh data/test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

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

mkdir -p ${SUNDA}

workspace=$PWD

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    cd ${SUNDA}
    idxs=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")
    for i in "${idxs[@]}"; do
        wget https://www.openslr.org/resources/36/asr_sundanese_${i}.zip
        unzip -o asr_sundanese_${i}.zip
        rm -f asr_sundanese_${i}.zip
    done
    mv asr_sundanese/* .
    rm -rf asr_sundanese
    cd $workspace
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/sunda_data_prep.py -d ${SUNDA}
    utils/spk2utt_to_utt2spk.pl data/sunda_train/spk2utt > data/sunda_train/utt2spk
    utils/spk2utt_to_utt2spk.pl data/sunda_dev/spk2utt > data/sunda_dev/utt2spk
    utils/spk2utt_to_utt2spk.pl data/sunda_test/spk2utt > data/sunda_test/utt2spk
    utils/fix_data_dir.sh data/sunda_train
    utils/fix_data_dir.sh data/sunda_dev
    utils/fix_data_dir.sh data/sunda_test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

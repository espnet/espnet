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

mkdir -p ${JAVA}

workspace=$PWD

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    cd ${JAVA}
    idxs=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")
    for i in "${idxs[@]}"; do
        wget https://www.openslr.org/resources/35/asr_javanese_${i}.zip
        unzip -o asr_javanese_${i}.zip
        rm -f asr_javanese_${i}.zip
    done
    mv asr_javanese/* .
    rm -rf asr_javanese
    cd $workspace
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/data_prep.py -d ${JAVA}
    utils/spk2utt_to_utt2spk.pl data/java_train/spk2utt > data/java_train/utt2spk
    utils/spk2utt_to_utt2spk.pl data/java_dev/spk2utt > data/java_dev/utt2spk
    utils/spk2utt_to_utt2spk.pl data/java_test/spk2utt > data/java_test/utt2spk
    utils/fix_data_dir.sh data/java_train
    utils/fix_data_dir.sh data/java_dev
    utils/fix_data_dir.sh data/java_test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

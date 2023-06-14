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

if [ -z "$MARATHI" ]; then
    log "Variable MARATHI not set in db.sh"
    exit 2
fi

mkdir -p ${MARATHI}

workspace=$PWD

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    cd ${MARATHI}
    wget https://www.openslr.org/resources/64/mr_in_female.zip
    unzip -o mr_in_female.zip
    rm -f mr_in_female.zip

    cd $workspace
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/data_prep.py -d ${MARATHI}
    utils/spk2utt_to_utt2spk.pl data/marathi_train/spk2utt > data/marathi_train/utt2spk
    utils/spk2utt_to_utt2spk.pl data/marathi_dev/spk2utt > data/marathi_dev/utt2spk
    utils/spk2utt_to_utt2spk.pl data/marathi_test/spk2utt > data/marathi_test/utt2spk
    utils/fix_data_dir.sh data/marathi_train
    utils/fix_data_dir.sh data/marathi_dev
    utils/fix_data_dir.sh data/marathi_test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

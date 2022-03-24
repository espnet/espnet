#!/bin/bash
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=1
SECONDS=0

. utils/parse_options.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "data preparation started"
if [ -z "${MARATHI}" ]; then
    log "Fill the value of 'MARATHI' in db.sh"
    exit 1
fi
mkdir -p ${MARATHI}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to ${MARATHI}"
    wget -O ${MARATHI}/mr_in_female.zip https://www.openslr.org/resources/64/mr_in_female.zip
    unzip -o -d ${MARATHI} ${MARATHI}/mr_in_female.zip
    rm -f ${MARATHI}/mr_in_female.zip    
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"
    python3 local/data_prep.py -d ${MARATHI}
    utils/spk2utt_to_utt2spk.pl data/train_mr/spk2utt > data/train_mr/utt2spk
    utils/spk2utt_to_utt2spk.pl data/dev_mr/spk2utt > data/dev_mr/utt2spk
    utils/spk2utt_to_utt2spk.pl data/test_mr/spk2utt > data/test_mr/utt2spk
    utils/fix_data_dir.sh data/train_mr
    utils/fix_data_dir.sh data/dev_mr
    utils/fix_data_dir.sh data/test_mr
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
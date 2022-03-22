#!/bin/bash


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

mkdir -p ${MALAYALAM}
if [ -z "${MALAYALAM}" ]; then
    log "Fill the value of 'MALAYALAM' of db.sh"
    exit 1
fi

workspace=$PWD

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    cd ${MALAYALAM}
    wget https://www.openslr.org/resources/63/ml_in_female.zip
    unzip -o ml_in_female.zip
    rm -f ml_in_female.zip
    wget https://www.openslr.org/resources/63/ml_in_male.zip
    unzip -o ml_in_male.zip
    rm -f ml_in_male.zip

    wget https://www.openslr.org/resources/63/line_index_female.tsv
    wget https://www.openslr.org/resources/63/line_index_male.tsv
    cat line_index_female.tsv line_index_male.tsv > line_index_all.tsv
    cd $workspace
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for openslr"

    python3 local/data_prep.py -d ${MALAYALAM}
    utils/spk2utt_to_utt2spk.pl data/train_ml/spk2utt > data/train_ml/utt2spk
    utils/spk2utt_to_utt2spk.pl data/dev_ml/spk2utt > data/dev_ml/utt2spk
    utils/spk2utt_to_utt2spk.pl data/test_ml/spk2utt > data/test_ml/utt2spk
    utils/fix_data_dir.sh data/train_ml
    utils/fix_data_dir.sh data/dev_ml
    utils/fix_data_dir.sh data/test_ml
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

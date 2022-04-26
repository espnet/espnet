#!/usr/bin/env bash

# Copyright 2021 Carnegie Mellon  University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
dsing=1

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${DSING}
if [ -z "${DSING}" ]; then
    log "Fill the value of 'DSING' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train${dsing}
train_dev=dev
test_set="dev test"


log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Prepare stage1: Download data to ${DSING}"
    echo "Please download the data at https://ccrma.stanford.edu/damp/"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Prepare stage2: segmentation setup for Dsing"
    if [ -d "local/dsing_task" ]; then
       echo "exist segmetation, skip git clone"
    else
        git clone https://github.com/groadabike/Kaldi-Dsing-task.git local/dsing_task
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Prepare stage3: dataset prepare"
    for datadir in ${train_set} ${train_dev} ${test_set}; do
        python local/data_prep.py data/ ${DSING}/sing_300x30x2 local/dsing_task/DSing\ Kaldi\ Recipe/dsing/s5/conf/${datadir}.json ${datadir}
        utils/utt2spk_to_spk2utt.pl data/${datadir}/utt2spk > data/${datadir}/spk2utt
        utils/fix_data_dir.sh data/${datadir}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

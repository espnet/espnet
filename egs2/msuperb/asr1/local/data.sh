#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
duration=10min # duration can be either 10min or 1h


 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MSUPERB}
if [ -z "${MSUPERB}" ]; then
    log "Fill the value of 'MSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_${duration}
train_dev=dev_${duration}
test_set=test_${duration}

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Download data to ${MSUPERB}"
    log "Not released yet"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for multilingual SUPERB"
    
    python data_prep.py \
        --train_set ${train_set} \
        --train_dev ${train_dev} \
        --test_set ${test_set} \
        --duration ${duration} \
        --source ${MSUPERB} \
        --lid false

    for x in ${train_set} ${train_dev} ${test_set}; do
        utils/fix_data_dir.sh data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data download
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${ASVTutorial}
if [ -z "${ASVTutorial}" ]; then
    log "Fill the value of 'ASVTutorial' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
train_dev="dev"
test_set="eval"

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage1: Download data to ${ASVTutorial}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for ASVTutorial"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "${train_set}" "${train_dev}" "${test_set}"; do
        # use underscore-separated names in data directories.
        python local/data_prep.py --src_folder "${ASVTutorial}" --subset ${part} --tgt data/${part}
        utils/utt2spk_to_spk2utt.pl data/${part}/utt2spk > data/${part}/spk2utt
        utils/fix_data_dir.sh data/${part}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

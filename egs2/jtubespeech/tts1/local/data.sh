#!/usr/bin/env bash

# Copyright 2021 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

ctcscore_pruning_threshold=-0.5

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${JTUBESPEECH}" ]; then
   log "Fill the value of 'JTUBESPEECH' of db.sh"
   exit 1
fi
db_root=${JTUBESPEECH}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage 0: Data Download"
    local/download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 1: local/data_prep.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    local/data_prep.sh "${db_root}/jtuberaw" "${db_root}/jtubesplit" data/train ${ctcscore_pruning_threshold}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --first data/train 500 data/devtest
    utils/subset_data_dir.sh --first data/devtest 250 data/dev
    utils/subset_data_dir.sh --last data/devtest 250 data/test
    n=$(( $(wc -l < data/train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

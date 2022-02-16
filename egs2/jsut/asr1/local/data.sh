#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1
fs=48000

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

db_root=${JSUT}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Initial normalization of the data
    local/data_prep.sh ${db_root}/jsut_ver1.1 data/train ${fs}
    utils/validate_data_dir.sh --no-feats data/train

    # changing the sampling rate option in pitch.conf and fbank.conf
    local/change_sampling_rate.sh ${fs}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --first data/train 500 data/deveval
    utils/subset_data_dir.sh --first data/deveval 250 data/${recog_set}
    utils/subset_data_dir.sh --last data/deveval 250 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

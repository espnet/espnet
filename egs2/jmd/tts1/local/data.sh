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
stop_stage=2
dialect="Kumamoto"

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${JMD}" ]; then
    log "Fill the value of 'JMD' of db.sh"
    exit 1
fi
db_root=${JMD}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

dialect=$(echo ${dialect} | tr '[:upper:]' '[:lower:]')

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}" "${dialect}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}" "${dialect}" data/train 24000
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --first data/train 100 data/deveval
    utils/subset_data_dir.sh --first data/deveval 50 data/${recog_set}
    utils/subset_data_dir.sh --last data/deveval 50 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 100 ))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

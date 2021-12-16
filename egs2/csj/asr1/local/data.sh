#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${CSJDATATOP}" ]; then
    log "Fill the value of 'CSJDATATOP' of db.sh"
    exit 1
fi
if [ -z "${CSJVER}" ]; then
    log "Fill the value of 'CSJVER' of db.sh"
    exit 1
fi

train_set=train_nodup
train_dev=train_dev
recog_set="eval1 eval2 eval3"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Initial normalization of the data
    local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} data/csj-data ${CSJVER}
    local/csj_data_prep.sh data/csj-data

    for x in ${recog_set}; do
        local/csj_eval_data_prep.sh data/csj-data/eval ${x}
    done

    for x in train eval1 eval2 eval3; do
        local/csj_rm_tag_sp_space.sh data/${x}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # make a development set
    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # 6hr 31min
    n=$(($(wc -l < data/train/segments) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

    # remove duplicated utterances in the training set
    utils/data/remove_dup_utts.sh 300 data/train_nodev data/${train_set} # 233hr 36min

fi

log "Successfully finished. [elapsed=${SECONDS}s]"

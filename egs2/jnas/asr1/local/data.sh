#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
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

train_set=train_nodev
train_dev=train_dev
recog_set="JNAS_testset_100 JNAS_testset_500"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Initial normalization of the data
    local/jnas_train_prep.sh ${jnas_train_root} ./conf/train_speakers.txt

    for rtask in ${recog_set}; do
        local/jnas_eval_prep.sh ${jnas_eval_root} ${rtask}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or 0 characters
    remove_longshortdata.sh data/train data/train_trim

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt.sh --perdt 5 data/train_trim \
            data/${train_set} data/${train_dev}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

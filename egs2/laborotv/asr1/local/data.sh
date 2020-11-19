#!/bin/bash

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

if [ ! -e "${LABOROTV}" ]; then
    log "Fill the value of 'LABOROTV' of db.sh"
    exit 1
fi

train_set=train_nodev
train_dev=train_dev

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Initial normalization of the data
    local/laborotv_data_prep.sh ${LABOROTV}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # make a development set during training by extracting the first 4000 utterances
    # followed by the CSJ recipe
    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # XXXhr XXmin
    n=$(($(wc -l < data/train/segments) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set} # XXXh XXXmin
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

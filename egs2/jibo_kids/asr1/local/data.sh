#!/usr/bin/env bash

# set -e
# set -u
# set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${JIBO_KIDS}" ]; then
    log "Fill the value of 'JIBO_KIDS' in db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${JIBO_KIDS}" ]; then
        log "stage 1: Please download data from https://github.com/balaji1312/Jibo_Kids and save to ${JIBO_KIDS}"
        exit 1
    else
        log "stage 1: ${JIBO_KIDS} already exists. Skipping data downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    original_dir="${JIBO_KIDS}"
    data_dir="./data"
    

    local/jibo_data_prepare.sh $original_dir/ $data_dir/
    log "stage 2: Data preparation completed."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Train Dev Test split"

    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 25 $data_dir/all \
    $data_dir/train $data_dir/test_dev

    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50 $data_dir/test_dev \
        $data_dir/test $data_dir/dev

    rm -rf $data_dir/test_dev

    log "Stage 3: Train Dev Test split completed."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

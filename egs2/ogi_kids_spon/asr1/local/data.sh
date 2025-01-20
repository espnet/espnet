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

if [ -z "${OGI_KIDS}" ]; then
    log "Fill the value of 'OGI_KIDS' in db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${OGI_KIDS}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2007S18 and save to ${OGI_KIDS}"
        exit 1
    else
        log "stage 1: ${OGI_KIDS} already exists. Skipping data downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    original_dir="${OGI_KIDS}"
    data_dir="./data"

    #First prepare all spon data without considering segment length
    local/ogi_spon_all_data_prepare.sh $original_dir/ $data_dir/
    log "stage 2: Data preparation completed."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: CTC Segment"

    #lists dir is used to ensure train, dev, test splits are done based on speakers from scripted ogi
    lists_dir="conf/file_list"

    python local/ctc_segment.py --input $data_dir/spont_all --lists $lists_dir --output $data_dir

    log "Stage 3: Finished ctc segmentation"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

#!/usr/bin/env bash

set -e
set -u
set -o pipefail

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

MYST=/ocean/projects/cis210027p/shared/corpora/ldc2021s05/

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${MYST}" ]; then
    log "Fill the value of 'MYST' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${MYST}/myst_child_conv_speech" ]; then
	    echo "stage 1: Download data to ${MYST}"
        exit 1
    else
        log "stage 1: ${MYST}/myst_child_conv_speech is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    # Set the base directory of the original dataset
    original_dir="${MYST}/myst_child_conv_speech/data"
    # Set the base directory for the target data
    data_dir="./data"

    python local/prepare_data.py --original-dir $original_dir --data-dir $data_dir
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
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
stop_stage=2

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

nlsyms_file=data/nlsyms.txt


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # create a dummy non-linguistic symbols file, these should be already removed in data prep
    log "stage 2: Create non linguistic symbols file: ${nlsyms_file}"
    touch ${nlsyms_file} # dummy empty file
fi

log "Successfully finished. [elapsed=${SECONDS}s]"


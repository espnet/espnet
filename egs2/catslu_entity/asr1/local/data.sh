#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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

. ./db.sh
. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CATSLU}" ]; then
    log "Fill the value of 'CATSLU' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${CATSLU}/catslu_traindev" ] && [ ! -e "${CATSLU}/catslu_test" ]; then
	echo "stage 1: Download traindev and test data to ${CATSLU}"
    else
        log "stage 1: ${CATSLU}/catslu_traindev and ${CATSLU}/catslu_test already exists."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    python3 local/data_prep.py ${CATSLU}
    for x in train devel test_map test_music test_video test_weather; do
    	utils/fix_data_dir.sh data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

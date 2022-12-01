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

if [ -z "${ELRA_E0024}" ]; then
    log "Fill the value of 'ELRA_E0024' of db.sh"
    exit 1
fi

if [ -z "${ELRA_S0272}" ]; then
    log "Fill the value of 'ELRA_S0272' of db.sh"
    exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 1: Data Preparation"
    python3 local/prepare_data.py ${ELRA_E0024} ${ELRA_S0272} data/
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

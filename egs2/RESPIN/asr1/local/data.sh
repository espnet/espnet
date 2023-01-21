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
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

        log "stage 1: done"
    
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	log "stage 2: Data Preparation: (completed beforehand)"

    
fi



log "Successfully finished. [elapsed=${SECONDS}s]"


  

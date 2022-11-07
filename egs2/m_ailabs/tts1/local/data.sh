#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

train_type=en_us
log "$0 $*"
. utils/parse_options.sh

if [ "${train_type}" = en_us ]; then
    # Training the model only on en_US
    local/data_en_us.sh
elif [ "${train_type}" = multilingual ]; then
    # Training the model on the whole dataset
    local/data_multilingual.sh
else
    log "train_type: ${train_type} is not supported."
    exit 1
fi

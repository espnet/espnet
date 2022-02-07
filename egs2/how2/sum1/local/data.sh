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
stop_stage=1

. ./db.sh
. ./path.sh
. ./cmd.sh

url_how2_2000=
data_how2=how2_feats

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ! -d ${data_how2_text} ]; then
    log "${data_how2_text} doesn't exist. Downloading from URL-----"
    wget 
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data download"

    if [ -d ${data_how2} ]; then
        log "$0: HowTo directory or archive already exists in ${data_how2_text}. Skipping download."
    else
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 2
        fi
        log "$0: downloading test set from ${url_iwslt19}"

        if ! wget --no-check-certificate -o how2_feats.tar.gz ${url_how2_2000}; then
            log "$0: error executing wget ${url_how2_2000}"
            exit 2
        fi

        if ! tar -xvzf how2_feats.tar.gz -C ${data_how2}; then
            log "$0: error un-tarring archive how2_feats.tar.gz"
            exit 2
        fi

        log "$0: Successfully downloaded and un-tarred how2_feats.tar.gz"
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation and verification"

fi 
log "Successfully finished. [elapsed=${SECONDS}s]"

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

url_how2_2000="https://drive.google.com/file/d/1SHg7La_hflMTIm6gaCus46sn4zYqWJvb/view?usp=sharing"
data_how2=how2_feats

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data download"

    if [ -d ${data_how2} ]; then
        log "$0: HowTo directory or archive already exists in ${data_how2}. Skipping download."
    else
        ../../../utils/download_from_google_drive.sh ${url_how2_2000} $PWD tar.gz
        log "$0: Successfully downloaded and un-tarred how2_feats.tar.gz"
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation and verification"
    mv how2_feats/data .
    mv how2_feats/fbank .
fi 

log "Successfully finished. [elapsed=${SECONDS}s]"

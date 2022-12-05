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

data_url=
data_how2=${HOW2_2kH}


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
        wget ${data_url} -o out.tar.bz2 
        tar -xvf out.tar.bz2 -C ${data_how2}
        log "$0: Successfully downloaded and un-tarred how2_feats"
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation and verification"

    mkdir -p data 
    for dir in tr_2000h_sum cv05_sum dev5_test_sum; do  
        [ -f data/${dir} ] || mv ${data_how2}/data/${dir} data/${dir}
        [ -f data/${dir}/feats.scp ] || awk -F ' ' -v x=$(realpath ${data_how2}) '{print $1,x"/audio/fbank_pitch/"$2}' < ${data_how2}/audio/fbank_pitch/${dir}.scp  > data/${dir}/feats.scp
        [ -f data/${dir}/wav.scp ] || cut -d ' ' -f2 data/${dir}/segments | sort | uniq | awk -F ' ' '{print $1,"<DUMMY>"}' > data/${dir}/wav.scp
        utils/fix_data_dir.sh data/${dir}
    done 
   
fi 

log "Successfully finished. [elapsed=${SECONDS}s]"

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
stop_stage=100

an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train"
train_dev="train_dev"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python3 local/data_prep.py ${an4_root} sph2pipe

    for x in test ${train_set}; do
        sort data/${x}/wav.scp -o data/${x}/wav.scp
    done

    # make a dev set
    mkdir -p data/${train_dev}
    cp -r data/train/wav.scp data/${train_dev}/wav.scp
fi


log "Successfully finished. [elapsed=${SECONDS}s]"

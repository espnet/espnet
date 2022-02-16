#!/usr/bin/env bash

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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

data_url=ftp://ftp.esat.kuleuven.be/psi/speech/vrenkens/grabo.tar.gz
data_tar=grabo.tar.gz
data_tar_size=2102543961

. ./db.sh
. ./path.sh
. ./cmd.sh

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z ${GRABO} ]; then
    log "Fill the value of 'GRABO' of db.sh"
    exit 1
fi

cur_path=$(pwd)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Download Data to ${GRABO}"
    if [ ! -d ${GRABO} ]; then
    mkdir -p ${GRABO}
    fi
    # absolute path
    GRABO=$(cd ${GRABO}; pwd)

    # download data files if they do not exist
    # file name: grabo.tar.gz
    download_data=true
    if [ -f ${GRABO}/${data_tar} ]; then
        size=$(/bin/ls -l ${GRABO}/${data_tar} | awk '{print $5}')
        if [ ${size} -eq ${data_tar_size} ]; then
            download_data=false
            log "${GRABO}/${data_tar} exists and appears to be complete."
        else
            log "$0: removing existing file ${GRABO}/${data_tar} because its size in bytes ${size} is not equal to the size of the archive."
            rm ${GRABO}/${data_tar}
        fi
    fi

    if ${download_data}; then
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 1
        fi

        cd ${GRABO}
        if ! wget --no-check-certificate ${data_url}; then
            log "$0: error executing wget ${data_url}"
            exit 1
        fi
    fi

    log "$0: successfully downloaded ${data_tar}"

    # un-tar
    cd ${GRABO}
    mkdir -p grabo
    if ! tar -xzf ${data_tar} -C grabo; then
        log "$0: error un-tarring archive ${data_tar}"
        exit 1
    fi

    log "$0: successfully un-tarred ${data_tar}"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    cd ${cur_path}
    mkdir -p data/{train,dev,test}
    python3 local/data_prep.py \
        --data_path ${GRABO}/grabo/grabo/speakers \
        --sox_path "$(command -v sox)" \
        --train_dir data/train \
        --dev_dir data/dev \
        --test_dir data/test
    for x in train dev test; do
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/fix_data_dir.sh data/${x}
        utils/validate_data_dir.sh --no-feats data/${x}
    done
fi

log "$0: successfully finished. [elapsed=${SECONDS}s]"

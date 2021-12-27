#!/usr/bin/env bash

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Google Speech Commands: https://arxiv.org/abs/1804.03209


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
num_commands=12         # 12 or 35

# data_url: the original location.
# test_data_url: a canonical test set for top-1 error.
data_url=http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
data_tar=speech_commands_v0.02.tar.gz
data_tar_size=2428923189
test_data_url=http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
test_data_tar=speech_commands_test_set_v0.02.tar.gz
test_data_tar_size=112563277


. ./db.sh
. ./path.sh
. ./cmd.sh

log "$0 $*"

. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z ${SPEECHCOMMANDS} ]; then
    log "Fill the value of 'SPEECHCOMMANDS' of db.sh"
    exit 1
fi

cur_path=$(pwd)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Download Data to ${SPEECHCOMMANDS}"
    if [ ! -d ${SPEECHCOMMANDS} ]; then
    mkdir -p ${SPEECHCOMMANDS}
    fi
    # absolute path
    SPEECHCOMMANDS=$(cd ${SPEECHCOMMANDS}; pwd)

    # download data files if they do not exist
    # file name: speech_commands_v0.02.tar.gz
    download_data=true
    if [ -f ${SPEECHCOMMANDS}/${data_tar} ]; then
        size=$(/bin/ls -l ${SPEECHCOMMANDS}/${data_tar} | awk '{print $5}')
        if [ ${size} -eq ${data_tar_size} ]; then
            download_data=false
            log "${SPEECHCOMMANDS}/${data_tar} exists and appears to be complete."
        else
            log "$0: removing existing file ${SPEECHCOMMANDS}/${data_tar} because its size in bytes ${size} is not equal to the size of the archive."
            rm ${SPEECHCOMMANDS}/${data_tar}
        fi
    fi

    if ${download_data}; then
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 1
        fi

        cd ${SPEECHCOMMANDS}
        if ! wget --no-check-certificate ${data_url}; then
            log "$0: error executing wget ${data_url}"
            exit 1
        fi
    fi

    # file name: speech_commands_test_set_v0.02.tar.gz
    download_test_data=true
    if [ -f ${SPEECHCOMMANDS}/${test_data_tar} ]; then
        size=$(/bin/ls -l ${SPEECHCOMMANDS}/${test_data_tar} | awk '{print $5}')
        if [ ${size} -eq ${test_data_tar_size} ]; then
            download_test_data=false
            log "${SPEECHCOMMANDS}/${test_data_tar} exists and appears to be complete."
        else
            log "$0: removing existing file ${SPEECHCOMMANDS}/${test_data_tar} because its size in bytes ${size} is not equal to the size of the archive."
            rm ${SPEECHCOMMANDS}/${test_data_tar}
        fi
    fi

    if ${download_test_data}; then
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 1
        fi

        cd ${SPEECHCOMMANDS}
        if ! wget --no-check-certificate ${test_data_url}; then
            log "$0: error executing wget ${test_data_url}"
            exit 1
        fi
    fi

    log "$0: successfully downloaded ${data_tar} and ${test_data_tar}"

    # un-tar
    cd ${SPEECHCOMMANDS}
    mkdir -p speech_commands_v0.02
    if ! tar -xzf ${data_tar} -C speech_commands_v0.02; then
        log "$0: error un-tarring archive ${data_tar}"
        exit 1
    fi

    mkdir -p speech_commands_test_set_v0.02
    if ! tar -xzf ${test_data_tar} -C speech_commands_test_set_v0.02; then
        log "$0: error un-tarring archive ${test_data_tar}"
        exit 1
    fi

    log "$0: successfully un-tarred ${data_tar} and ${test_data_tar}"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    cd ${cur_path}

    # prepare datasets
    if [ ${num_commands} -eq 12 ]; then
        log "Using 12 commands"
        mkdir -p data/{train,dev,test,test_speechbrain}
        python3 local/data_prep_12.py \
            --data_path ${SPEECHCOMMANDS}/speech_commands_v0.02 \
            --test_data_path ${SPEECHCOMMANDS}/speech_commands_test_set_v0.02 \
            --train_dir data/train \
            --dev_dir data/dev \
            --test_dir data/test \
            --speechbrain_testcsv local/speechbrain_test.csv \
            --speechbrain_test_dir data/test_speechbrain
        for x in train dev test test_speechbrain; do
            utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
            utils/fix_data_dir.sh data/${x}
            utils/validate_data_dir.sh --no-feats data/${x}
        done
    elif [ ${num_commands} -eq 35 ]; then
        log "Using 35 commands"
        mkdir -p data/{train,dev,test}
        python3 local/data_prep_35.py \
            --data_path ${SPEECHCOMMANDS}/speech_commands_v0.02 \
            --train_dir data/train \
            --dev_dir data/dev \
            --test_dir data/test
        for x in train dev test; do
            utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
            utils/fix_data_dir.sh data/${x}
            utils/validate_data_dir.sh --no-feats data/${x}
        done
    else
        log "$0: invalid num_commands: ${num_commands}"
        exit 1
    fi
fi

log "$0: successfully finished. [elapsed=${SECONDS}s]"

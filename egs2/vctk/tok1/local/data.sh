#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=-1
stop_stage=0

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are supported"
    exit 2
fi

. ./path.sh
. ./cmd.sh
. ./db.sh

if [ -z "${VCTK}" ]; then
    log "Error: Set VCTK in db.sh to a download root or existing corpus root"
    exit 1
fi

download_root=${VCTK}
if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    log "Stage -1: Locate VCTK or download VCTK 0.92"
    local/data_download.sh "${download_root}"
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    if [ -d "${download_root}/wav48_silence_trimmed" ] \
            || [ -d "${download_root}/wav48" ]; then
        corpus_dir=${download_root}
    elif [ -d "${download_root}/VCTK-Corpus-0.92/wav48_silence_trimmed" ]; then
        corpus_dir=${download_root}/VCTK-Corpus-0.92
    elif [ -d "${download_root}/VCTK-Corpus/wav48" ]; then
        corpus_dir=${download_root}/VCTK-Corpus
    else
        log "Error: No supported VCTK layout was found under ${download_root}"
        exit 1
    fi

    log "Stage 0: Prepare VCTK data from ${corpus_dir}"
    local/data_prep.sh \
        --train_set tr_no_dev \
        --dev_set dev \
        --eval_set eval1 \
        --microphone mic1 \
        "${corpus_dir}"
fi

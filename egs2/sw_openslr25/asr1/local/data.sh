#!/usr/bin/env bash

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Data preparation for the ALFFA Swahili broadcast-news ASR corpus (OpenSLR SLR25).

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

# general configuration
stage=0        # start from 0 if you need to start from data download
stop_stage=100
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

set -e
set -u
set -o pipefail

. utils/parse_options.sh

log "data preparation started"

if [ -z "${ALFFA}" ]; then
    log "Fill the value of 'ALFFA' in db.sh"
    exit 1
fi

corpus_dir="${ALFFA}/swahili"
tarball="data_broadcastnews_sw.tar.bz2"
url="https://www.openslr.org/resources/25/${tarball}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download data to ${corpus_dir}"
    mkdir -p "${corpus_dir}"
    if [ -e "${corpus_dir}/.complete" ]; then
        log "Already downloaded and extracted. Skipping (delete ${corpus_dir}/.complete to redo)."
    else
        wget -O "${corpus_dir}/${tarball}" "${url}"
        tar -xjf "${corpus_dir}/${tarball}" -C "${corpus_dir}"
        touch "${corpus_dir}/.complete"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing data directories (sw_train / sw_dev / sw_test)"
    python3 local/data_prep.py -d "${corpus_dir}" -o data

    for x in sw_train sw_dev sw_test; do
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/fix_data_dir.sh data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

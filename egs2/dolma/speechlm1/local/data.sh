#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Prepare text LM training corpus Dolma into Espnet format
# Dataset website: https://huggingface.co/datasets/allenai/dolma

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
stop_stage=3
DATA_DIR="./data/dolma"
PARALLEL_DOWNLOADS="16"
DOLMA_VERSION="v1_6-sample"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "download dataset"
    git clone https://huggingface.co/datasets/allenai/dolma
    mkdir -p "${DATA_DIR}"
    cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "file formatting"
    all_files=`find ${DATA_DIR} -name *.json.gz`
    for file in ${all_files}; do
        file_name=$(basename ${file})
        clean_name=${file_name%.json.gz}
        gunzip -c ${file} | jq -r '.text | @json' | sed 's/^"\(.*\)"$/\1/' |\
            awk -v prefix="$clean_name" \
            'BEGIN {srand()} {if (NF <= 500) {print prefix "_" NR " " $0}}' \
            > $(dirname ${file})/${clean_name} &
    done; wait

    mkdir -p data/train
    rm -f data/train/text
    for file in `ls ${DATA_DIR} | grep -v json.gz`; do
        cat ${DATA_DIR}/${file} >> data/train/text
    done
fi

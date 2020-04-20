#!/bin/bash
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

train_set=train_960
train_dev=dev

stage=1
stop_stage=100

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data
data_url=www.openslr.org/resources/12

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    mkdir -p ${datadir}
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LM external data preparation"
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        mkdir -p data/local/lm_train
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    zcat data/local/lm_train/librispeech-lm-norm.txt.gz | sed '/^$/d' | \
        awk '{ printf("librispeech_lm_%07d %s\n", NR, $0) }' \
        > data/local/lm_train/librispeech-lm-norm.txt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

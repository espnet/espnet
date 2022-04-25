#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

# base url for downloads.
data_url=https://us.openslr.org/resources/108/FR.tgz

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MEDIASPEECH}
if [ -z "${MEDIASPEECH}" ]; then
    log "Fill the value of 'MEDIASPEECH' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Download data to ${MEDIASPEECH}"
    log "The default data of this recipe is from mediaspeech - french"
    local/download_and_untar.sh ${MEDIASPEECH} ${data_url} FR.tgz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for mediaspeech"
    ### Task dependent. You have to make data the following preparation part by yourself.
    mkdir -p data/{train_as,dev_as,test_as,validated_as}
    python3 local/data_prep.py \
        --data_path ${MEDIASPEECH}/FR \
        --train_dir data/train_as \
        --dev_dir data/dev_as \
        --test_dir data/test_as \
        --validated_dir data/validated_as \
        --dev_ratio 0.1 \
        --test_ratio 0.1 \
        --validated_ratio 0.01
    for x in train_as dev_as test_as validated_as; do
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/fix_data_dir.sh data/${x}
        utils/validate_data_dir.sh --no-feats data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

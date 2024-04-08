#!/bin/bash

# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${TOTONAC}
if [ -z "${TOTONAC}" ]; then
    log "Fill the value of 'TOTONAC' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

wavdir=${TOTONAC}
annotation_dir=${TOTONAC}

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage1: Download data to ${TOTONAC}"
    mkdir -p ${TOTONAC}
    wget --no-check-certificate --directory-prefix=${TOTONAC} https://www.openslr.org/resources/107/Amith-Lopez_Totonac-recordings-northern-Puebla-and-adjacent-Veracruz_Metadata.xml
    local/download_and_untar.sh ${TOTONAC} https://www.openslr.org/resources/107/Totonac_Corpus.tgz Totonac_Corpus.tgz
    git clone https://github.com/ftshijt/Totonac_Split.git local/split
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for TOTONAC"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for x in train dev "test"; do
        python local/data_prep.py -w $wavdir -t data/${x} -i local/split/${x}.tsv -a ${annotation_dir}
        # sort -o data/${x}/utt2spk > data/${x}/utt2spk
        utils/fix_data_dir.sh data/${x}
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"

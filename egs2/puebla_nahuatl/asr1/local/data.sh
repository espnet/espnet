#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

# dataset related
annotation_type=eaf

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${PUEBLA_NAHUATL}
if [ -z "${PUEBLA_NAHUATL}" ]; then
    log "Fill the value of 'PUEBLA_NAHUATL' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

wavdir=${PUEBLA_NAHUATL}/Sound-files-Puebla-Nahuatl
annotation_dir=local/Pueble-Nahuatl-Manifest

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
    log "stage1: Download data to ${PUEBLA_NAHUATL}"
    mkdir -p ${PUEBLA_NAHUATL}
    local/download_and_untar.sh local  https://www.openslr.org/resources/92/Puebla-Nahuatl-Manifest.tgz Puebla-Nahuatl-Manifest.tgz
    local/download_and_untar.sh ${PUEBLA_NAHUATL} https://www.openslr.org/resources/92/Sound-Files-Puebla-Nahuatl.tgz.part0 Sound-Files-Puebla-Nahuatl.tgz.part0 9
    git clone https://github.com/ftshijt/Puebla_Nahuatl_Split.git local/split
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for Puebla Nahuatl"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for x in train dev "test"; do
        python local/data_prep.py -w $wavdir -t data/${x} -m ${annotation_type} -i local/split/speaker_wav_mapping_nahuatl_${x}.csv -a ${annotation_dir}
        utils/fix_data_dir.sh data/${x}
        chmod +x data/${x}/remix_script.sh
        mkdir -p remixed
        ./data/${x}/remix_script.sh
    done
    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

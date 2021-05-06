#!/bin/bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
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
annotation_id=mixtec_underlying_full
text_format=underlying_full # surface, underlying_reduce

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${YOLOXOCHITL_MIXTEC}
if [ -z "${YOLOXOCHITL_MIXTEC}" ]; then
    log "Fill the value of 'YOLOXOCHITL_MIXTEC' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_new
train_dev=dev_new
test_set=test_new


YOLOXOCHITL_MIXTEC=/export/c04/jiatong/data/
wavdir=${YOLOXOCHITL_MIXTEC}/Yoloxochitl-Mixtec-for-ASR/Sound-files-Narratives-for-ASR
annodir=${YOLOXOCHITL_MIXTEC}/Yoloxochitl-Mixtec-for-ASR/Transcriptions-for-ASR/ELAN-files-with-underlying-and-surface-tiers

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Download data to ${YOLOXOCHITL_MIXTEC}"
    # mkdir -p ${YOLOXOCHITL_MIXTEC}
    # local/download_and_untar.sh ${YOLOXOCHITL_MIXTEC} http://www.openslr.org/resources/89/Yoloxochitl-Mixtec-Data.tgz Yoloxochitl-Mixtec-Data.tgz
    # local/download_and_untar.sh local http://www.openslr.org/resources/89/Yoloxochitl-Mixtec-Manifest.tgz Yoloxochitl-Mixtec-Manifest.tgz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Preparing data for yoloxochitl_mixtec"
    python3 local/data_prep.py -w $wavdir -a $annodir -t data/${annotation_id} \
                              -m ${annotation_type} -i local/speaker_wav_mapping_mixtec_remove_reserve.csv \
                              -f ${text_format}
    chmod +x data/${annotation_id}/remix_script.sh
    mkdir -p remixed
    # ./data/${annotation_id}/remix_script.sh

    # ESPNet Version (same as voxforge)
    # consider duplicated sentences (does not consider speaker split)
    # filter out the same sentences (also same text) of test&dev set from validated set
    local/split_tr_dt_et.sh data/${annotation_id} data/${train_set} data/${train_dev} data/${test_set}
    
    # add speed perturbation
    train_set_org=${train_set}
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set_org} data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set_org} data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set_org} data/temp3
    train_set=${train_set_org}_sp
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

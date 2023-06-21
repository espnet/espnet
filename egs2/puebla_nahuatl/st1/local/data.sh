#!/usr/bin/env bash

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

# dataset related
wavdir=${PUEBLA_NAHUATL}/Sound-files-Puebla-Nahuatl
annotation_dir=${PUEBLA_NAHUATL}/SpeechTranslation210217
annotation_type=eaf
annotation_id=st
src_lang=na
tgt_lang=es

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data downloading"
    # Download the Data
    local/download_and_untar.sh local  https://www.openslr.org/resources/92/Puebla-Nahuatl-Manifest.tgz Puebla-Nahuatl-Manifest.tgz
    local/download_and_untar.sh ${PUEBLA_NAHUATL} https://www.openslr.org/resources/92/Sound-Files-Puebla-Nahuatl.tgz.part0 Sound-Files-Puebla-Nahuatl.tgz.part0 9
    local/download_and_untar.sh ${PUEBLA_NAHUATL} https://www.openslr.org/resources/92/SpeechTranslation_Nahuatl_Manifest.tgz SpeechTranslation_Nahuatl_Manifest.tgz
    git clone https://github.com/ftshijt/Puebla_Nahuatl_Split.git local/split
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"
    mkdir -p remixed
    for x in train dev test; do
        python local/data_prep.py -w $wavdir -t data/${x}_${annotation_id} -m ${annotation_type} -i local/split/speaker_wav_mapping_nahuatl_${x}.csv -a ${annotation_dir} -d local/split/Puebla-Nahuat-and-Totonac-consultants_for-LDC-archive.xml
        cp data/${x}_${annotation_id}/text.${src_lang} data/${x}_${annotation_id}/text.lc.rm.${src_lang}
        cp data/${x}_${annotation_id}/text.${tgt_lang} data/${x}_${annotation_id}/text.lc.rm.${tgt_lang}
        ln -sf data/${x}_${annotation_id}/text.lc.rm.${tgt_lang} data/${x}_${annotation_id}/text
        utils/fix_data_dir.sh --utt_extra_files "text.${src_lang} text.${tgt_lang} text.lc.rm.${src_lang} text.lc.rm.${tgt_lang}" data/${x}_${annotation_id}
        # shellcheck disable=SC1090
        . ./data/${x}_st/remix_script.sh
    sort -o data/${x}_${annotation_id}/text.lc.rm.${tgt_lang} data/${x}_${annotation_id}/text.lc.rm.${tgt_lang}
    sort -o data/${x}_${annotation_id}/text.lc.rm.${src_lang} data/${x}_${annotation_id}/text.lc.rm.${src_lang} 
    done
fi

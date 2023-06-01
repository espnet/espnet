#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100
fs=None
g2p=pyopenjtalk


log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${KIRITAN}" ]; then
    log "Fill the value of 'KIRITAN' of db.sh"
    exit 1
fi

mkdir -p ${KIRITAN}

train_set="tr_no_dev"
train_dev="dev"
recog_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The kiritan data should be downloaded from https://zunko.jp/kiridev/login.php
    # with authentication

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    # We use a pre-defined split (see details in local/dataset_split.py)"
    python local/dataset_split.py ${KIRITAN} \
        data/${train_set} data/${train_dev} data/${recog_set} --fs ${fs}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare segments and align"
    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}
        local/prep_segments_from_xml.py --silence P --silence B ${src_data}
        mv ${src_data}/text.tmp ${src_data}/text
        mv ${src_data}/segments_from_xml.tmp ${src_data}/segments_from_xml
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        local/prep_segments.py --g2p ${g2p} \
            --silence pau --silence sil --silence br ${src_data}
        mv ${src_data}/segments.tmp ${src_data}/segments
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        awk '{printf("%s kiritan\n", $1);}' < ${src_data}/segments > ${src_data}/utt2spk
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${src_data}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

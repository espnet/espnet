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
fs=24000
g2p=None

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${M4SINGER}" ]; then
    log "Fill the value of 'M4SINGER' of db.sh"
    exit 1
fi

mkdir -p ${M4SINGER}

train_set="tr_no_dev"
train_dev="dev"
recog_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The M4SINGER data should be downloaded from https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view?usp=share_link
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p wav_dump
    # we convert the music score to xml format
    python local/data_prep.py ${M4SINGER} \
        --wav_dumpdir wav_dump \
        --sr ${fs} \
        --g2p ${g2p}
    for src_data in ${train_set} ${train_dev} ${recog_set}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${src_data}
    done
fi

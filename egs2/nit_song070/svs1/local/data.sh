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

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${NIT_SONG070}" ]; then
    log "Fill the value of 'NIT_SONG070' of db.sh"
    exit 1
fi

mkdir -p ${NIT_SONG070}

train_set="train"
train_dev="dev"
eval_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The nit data should be downloaded from http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2
    # Terms from http://hts.sp.nitech.ac.jp/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p score_dump
    mkdir -p wav_dump
    python local/data_prep.py ${NIT_SONG070}/HTS-demo_NIT-SONG070-F001/data --midi_note_scp local/midi-note.scp \
        --score_dump score_dump \
        --wav_dumpdir wav_dump \
        --sr ${fs}
    for src_data in ${train_set} ${train_dev} ${eval_set}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${src_data}
    done
fi

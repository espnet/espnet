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
    log "Fill the value of 'HIT_SONG070' of db.sh"
    exit 1
fi

mkdir -p ${NIT_SONG070}

train_set=train
train_dev=dev
eval_set=eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The nit data should be downloaded from http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2
    # Terms from http://hts.sp.nitech.ac.jp/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p midi_dump
    mkdir -p wav_dump
    # we convert the music score to midi format
    python local/data_prep.py ${NIT_SONG070}/HTS-demo_NIT-SONG070-F001/data --midi_note_scp local/midi-note.scp \
        --midi_dumpdir midi_dump \
        --wav_dumpdir wav_dump \
        --sr ${fs}
    for src_data in ${train_set} ${train_dev} ${eval_set}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label midi.scp" data/${src_data}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare segments"
    for dataset in ${train_set} ${train_dev} ${eval_set}; do
        src_data=data/${dataset}
        local/prep_segments.py --silence pau --silence sil ${src_data} 10000 # in ms
        mv ${src_data}/segments.tmp ${src_data}/segments
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/text.tmp ${src_data}/text
	cat ${src_data}/segments | awk '{printf("%s nit_song070\n", $1);}' > ${src_data}/utt2spk
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files label ${src_data}
    done
fi

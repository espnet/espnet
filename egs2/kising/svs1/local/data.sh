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
fs=44100
g2p=None
dataset='all'

train_set="tr_no_dev"
train_dev="dev"

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${KISING}" ]; then
    log "Fill the value of 'KISING' of db.sh"
    exit 1
fi

mkdir -p ${KISING}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    log "The KISING data should be downloaded"

    log "automatically download from google drive"
    ./local/download_wget.sh "1Wi8luF2QF6jsXYnO78uiWqZGJrtGuXb_"  ${KISING}/kising-v2.zip
    ./local/download_wget.sh "1VX8Fbu-Etv94LZHx928VJ12jfP8MEBNz"  ${KISING}/kising-v2-original.zip

    unzip ${KISING}/kising-v2.zip -d ${KISING}/KISING
    unzip ${KISING}/kising-v2-original.zip -d ${KISING}/KISING
    mv ${KISING}/clean ${KISING}/original
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p wav_dump
    # we convert the music score to xml format
    python local/data_prep.py ${KISING}/KISING --midi_note_scp local/midi-note.scp \
        --wav_dumpdir wav_dump \
        --sr ${fs} \
        --g2p ${g2p}\
        --dataset ${dataset}
    for src_data in train_${dataset} test_${dataset}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${src_data}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Held out validation set"

    utils/copy_data_dir.sh data/train data/${train_set}
    utils/copy_data_dir.sh data/train data/${train_dev}
    for dset in ${train_set} ${train_dev}; do
        for extra_file in label score.scp; do
            cp data/train/${extra_file} data/${dset}
        done
    done
    tail -n 50 data/train_${dataset}/wav.scp > data/${train_dev}/wav.scp
    utils/filter_scp.pl --exclude data/dev/wav.scp data/train_${dataset}/wav.scp > data/${train_set}/wav.scp

    utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${train_set}
    utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${train_dev}

fi

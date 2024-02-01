#!/usr/bin/env bash

# Reference from ESPnet's egs2/nit_song070/svs1/local/data.sh
# https://github.com/espnet/espnet/blob/master/egs2/nit_song070/svs1/local/data.sh


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
# shellcheck disable=SC2034
g2p=None

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${JSUT_SONG}" ]; then
    log "Fill the value of 'JSUT_SONG' of db.sh"
    exit 1
fi

mkdir -p ${JSUT_SONG}

train_set="train"
train_dev="dev"
eval_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The jsut-song data should be downloaded from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song
    # Terms from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song

    # Please ensure that you've downloaded songs (jsut-song_ver1.zip) and labels (jsut-song_label.zip) to ${JSUT_SONG} before proceeding
    unzip ${JSUT_SONG}/jsut-song_ver1.zip -d ${JSUT_SONG}
    unzip ${JSUT_SONG}/jsut-song_label.zip -d ${JSUT_SONG}
    rm ${JSUT_SONG}/jsut-song_ver1.zip
    rm ${JSUT_SONG}/jsut-song_label.zip
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p score_dump
    mkdir -p wav_dump
    python local/data_prep.py \
        --lab_srcdir ${JSUT_SONG}/todai_child \
        --wav_srcdir ${JSUT_SONG}/jsut-song_ver1/child_song/wav \
        --score_dump score_dump \
        --wav_dumpdir wav_dump \
        --sr ${fs}
    for src_data in ${train_set} ${train_dev} ${eval_set}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${src_data}
    done
fi

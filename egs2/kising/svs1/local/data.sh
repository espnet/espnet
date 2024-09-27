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
# shellcheck disable=SC2034
g2p=None
dataset="all"

train_set="tr_no_dev"
valid_set="dev"
test_set="eval"

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
    # ./local/download_google_drive.sh "1Wi8luF2QF6jsXYnO78uiWqZGJrtGuXb_" "${KISING}/kising-v2.zip"
    # ./local/download_google_drive.sh "1VX8Fbu-Etv94LZHx928VJ12jfP8MEBNz" "${KISING}/kising-v2-original.zip"

    # unzip "${KISING}/kising-v2.zip" -d "${KISING}/KISING"
    # unzip "${KISING}/kising-v2-original.zip" -d "${KISING}/KISING"
    # mv "${KISING}/KISING/clean" "${KISING}/KISING/original"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p "${KISING}/KISING/all"

    # # Resample files to sampling rate fs, single channel, 16 bits
    # for song_folder in "${KISING}/KISING/kising-v2"/*; do
    #     # Skip if song_folder ends with -unseg or is between 436 and 440
    #     if [[ "${song_folder}" == *-unseg ]]; then
    #         continue
    #     fi
    #     song_id=$(basename "${song_folder}")
    #     song="${song_id%%-*}"
    #     if [[ "${song}" -ge 436 ]] && [[ "${song}" -le 440 ]]; then
    #         continue
    #     fi
    #     mkdir -p "${KISING}/KISING/all/${song_id}"
    #     for file in "${song_folder}"/*.wav; do
    #         filename=$(basename ${file})
    #         sox ${file} -r ${fs} -b 16 -c 1 "${KISING}/KISING/all/${song_id}/${filename}"
    #     done
    #     cp -r "${song_folder}"/*.mid "${KISING}/KISING/all/${song_id}"
    # done
    # for file in "${KISING}/KISING/original"/*.wav; do
    #     filename=$(basename ${file})
    #     song_id=$(echo ${filename} | cut -c 1-3) # e.g., 421_all.wav -> 421
    #     if [[ ${filename} == *part2.wav ]]; then
    #         song_id=${song_id}-2 # e.g., 441-unseg-part2.wav -> 441-2
    #     fi
    #     mkdir -p "${KISING}/KISING/all/${song_id}"
    #     sox ${file} -r ${fs} -b 16 -c 1 "${KISING}/KISING/all/${song_id}/${song_id}-original.wav"
    # done

    mkdir -p wav_dump
    # we convert the music score to xml format
    python local/data_prep.py "${KISING}/KISING/all" \
        --wav_dumpdir wav_dump \
        --dataset ${dataset}
    for src_data in train test; do
        utils/utt2spk_to_spk2utt.pl <"data/${src_data}_${dataset}/utt2spk" >"data/${src_data}_${dataset}/spk2utt"
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" "data/${src_data}_${dataset}"
    done
    if [ -e "data/${test_set}" ]; then
        rm -r "data/${test_set}"
    fi
    mv "data/test_${dataset}" data/${test_set}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Held out validation set"

    utils/copy_data_dir.sh "data/train_${dataset}" "data/${train_set}"
    utils/copy_data_dir.sh "data/train_${dataset}" "data/${valid_set}"
    for dset in ${train_set} ${valid_set}; do
        for extra_file in label score.scp; do
            cp "data/train_${dataset}/${extra_file}" "data/${dset}"
        done
    done
    tail -n 50 "data/train_${dataset}/wav.scp" >"data/${valid_set}/wav.scp"
    utils/filter_scp.pl --exclude data/dev/wav.scp "data/train_${dataset}/wav.scp" >"data/${train_set}/wav.scp"

    utils/fix_data_dir.sh --utt_extra_files "label score.scp" "data/${train_set}"
    utils/fix_data_dir.sh --utt_extra_files "label score.scp" "data/${valid_set}"

fi

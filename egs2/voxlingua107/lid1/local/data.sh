#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100000
train_set="train_voxlingua107"
dev_set="dev_voxlingua107"

log "$0 $*"
. utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${VOXLINGUA107}" ]; then
    log "Fill the value of 'VOXLINGUA107' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Download"
    mkdir -p "${VOXLINGUA107}"
    wget -P "${VOXLINGUA107}" https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/zip_urls.txt
    xargs wget --continue -P "${VOXLINGUA107}" < "${VOXLINGUA107}/zip_urls.txt"
    find "${VOXLINGUA107}" -type f -name "*.zip" | while read -r zip_file; do
        if [[ "$(basename "${zip_file}")" == "dev.zip" ]]; then
            log "Extracting dev.zip to ${VOXLINGUA107}/dev"
            unzip -q -o "${zip_file}" -d "${VOXLINGUA107}/dev"
        else
            log "Extracting ${zip_file} to ${VOXLINGUA107}"
            unzip -q -o "${zip_file}" -d "${VOXLINGUA107}"
        fi
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generate wav.scp for train and dev sets"
    mkdir -p data
    mkdir -p data/${train_set}
    mkdir -p data/${dev_set}
    find "${VOXLINGUA107}" -type f -name "*.wav" | grep -v "/dev/" | awk -F '/' '{print $NF, $0}' > data/${train_set}/wav.scp
    find "${VOXLINGUA107}/dev" -type f -name "*.wav" | awk -F '/' '{print $NF, $0}' > data/${dev_set}/wav.scp
    python local/prepare_voxlingua107.py --func_name gen_wav_scp
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generate utt2lang for train and dev sets"
    python local/prepare_voxlingua107.py --func_name gen_utt2lang
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Convert utt2lang to lang2utt and create category2utt"
    utils/utt2spk_to_spk2utt.pl data/${train_set}/utt2lang > data/${train_set}/lang2utt
    utils/utt2spk_to_spk2utt.pl data/${dev_set}/utt2lang > data/${dev_set}/lang2utt

    log "Copying lang2utt to category2utt for train and dev sets"
    cp data/${train_set}/lang2utt data/${train_set}/category2utt
    cp data/${dev_set}/lang2utt data/${dev_set}/category2utt
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Fix and validate data directories"
    # Temporarily copy utt2lang to utt2spk and lang2utt to spk2utt
    # Because fix_data_dir.sh and validate_data_dir.sh expect these files
    cp data/${train_set}/utt2lang data/${train_set}/utt2spk
    cp data/${dev_set}/utt2lang data/${dev_set}/utt2spk
    cp data/${train_set}/lang2utt data/${train_set}/spk2utt
    cp data/${dev_set}/lang2utt data/${dev_set}/spk2utt

    utils/fix_data_dir.sh data/${train_set} || exit 1
    utils/fix_data_dir.sh data/${dev_set} || exit 1
    utils/validate_data_dir.sh --no-feats --non-print --no-text data/${train_set} || exit 1
    utils/validate_data_dir.sh --no-feats --non-print --no-text data/${dev_set} || exit 1

    mv data/${train_set}/utt2spk data/${train_set}/utt2lang
    mv data/${dev_set}/utt2spk data/${dev_set}/utt2lang
    mv data/${train_set}/spk2utt data/${train_set}/lang2utt
    mv data/${dev_set}/spk2utt data/${dev_set}/lang2utt
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${VOXLINGUA107}/rirs_noises.zip ]; then
        wget -P ${VOXLINGUA107} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${VOXLINGUA107}/musan.tar.gz ]; then
        wget -P ${VOXLINGUA107} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${VOXLINGUA107}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${VOXLINGUA107}/rirs_noises.zip -d ${VOXLINGUA107}
    fi

    if [ -d ${VOXLINGUA107}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${VOXLINGUA107}/musan.tar.gz -C ${VOXLINGUA107}
    fi

    # make scp files
    for x in music noise speech; do
        find ${VOXLINGUA107}/musan/${x} -iname "*.wav" > data/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    find ${VOXLINGUA107}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > data/rirs.scp
    find ${VOXLINGUA107}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> data/rirs.scp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

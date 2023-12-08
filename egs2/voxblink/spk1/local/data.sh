#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
n_proc=8

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



if [ -z ${VOXBLINK} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/voxblink/spk1/downloads"
    VOXBLINK=${MAIN_ROOT}/egs2/voxblink
else
    log "Root dir is: ${VOXBLINK}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Clone VoxBlink downloader and download essential files"
    log "folder names 'resource' should exist in egs2/voxblink/spk1"


    # clone downloader repo
    git clone https://github.com/VoxBlink/ScriptsForVoxBlink ${VOXBLINK}/downloader_repo

    if [ -d resource ]; then
        mv resource ${VOXBLINK}/downloader_repo
    else
        log "Make sure you download 'resources' from the VoxBlink organizers after requesting access to their cloud"
    fi

    org_dir=$(pwd)
    cd ${VOXBLINK/downloader_repo/resource}

    #unzip timestamp and video tags
    tar -zxvf timestamp.tar.gz
    tar -zxvf video_tags.tar.gz

    if [ ! -x ffmpeg ]; then
        log "ffmpeg is required"
        exit 1
    fi

    cd ..
    python3 -m pip install -r requirements.txt
    cd ${org_dir}

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Crawl VoxBlink from YouTube"

    # download YouTube videos
    cd ${VOXBLINK}/downloader_repo
    python3 downloader.py --base_dir ${VOXBLINK}/videos --num_workers ${n_proc} --mode full

    # crop audios
    python3 cropper.py --save_dir ${VOXBLINK} --timestamp_dir ${VOXBLINK}/downloader_repo/resource/timestamp --num_workers ${n_proc} --mode full --video_dir ${VOXBLINK}/videos

    cd ${MAIN_ROOT}/egs2/voxblink/spk1

    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${VOXBLINK}/rirs_noises.zip ]; then
        wget -P ${VOXBLINK} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${VOXBLINK}/musan.tar.gz ]; then
        wget -P ${VOXBLINK} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${VOXBLINK}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${VOXBLINK}/rirs_noises.zip -d ${VOXBLINK}
    fi

    if [ -d ${VOXBLINK}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${VOXBLINK}/musan.tar.gz -C ${VOXBLINK}
    fi

    # make scp files
    for x in music noise speech; do
        find ${VOXBLINK}/musan/${x} -iname "*.wav" > ${VOXBLINK}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    find ${VOXBLINK}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${VOXBLINK}/rirs.scp
    find ${VOXBLINK}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${VOXBLINK}/rirs.scp
    log "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Change into kaldi-style feature."
    log "This recipe uses VoxCeleb1 test set following official VoxBlink paper."
    log "Run stage 1 of egs2/voxceleb/spk1/spk.sh first"

    mkdir -p ${trg_dir}/voxblink_full
    python local/data_prep.py --src "${VOXBLINK}/audio" --dst "${trg_dir}/voxblink_full"

    for f in wav.scp utt2spk spk2utt; do
        sort ${trg_dir}/voxblink_full/${f} -o ${trg_dir}/voxblink_full/${f}
    done

    # make a symlink of VoxCeleb1 test set
    ln -s ${MAIN_ROOT}/egs2/voxceleb/spk1/data/voxceleb1_test ${trg_dir}/voxceleb1_test
    log "Stage 4, DONE."
fi

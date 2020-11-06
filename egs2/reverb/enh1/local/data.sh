#!/bin/bash

# Copyright 2018  Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh


stage=1
stop_stage=2
wavdir=${PWD}/wav # set the directory of the multi-condition training WAV files to be generated
nch_train=2 # number of training channels

. utils/parse_options.sh


if [ ! -e "${REVERB}" ]; then
    log "Fill the value of 'REVERB' of db.sh"
    exit 1
fi
if [ ! -e "${WSJCAM0}" ]; then
    log "Fill the value of 'WSJCAM0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data simulation"

    local/generate_data.sh --wavdir ${wavdir} ${WSJCAM0}
    local/prepare_simu_data.sh --wavdir ${wavdir} ${REVERB} ${WSJCAM0}
    local/prepare_real_data.sh --wavdir ${wavdir} ${REVERB}

    # Run WPE and Beamformit
    local/run_wpe.sh
    local/run_beamform.sh ${wavdir}/WPE/
    # Download and install speech enhancement evaluation tools
    if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
        # download and install speech enhancement evaluation tools
        local/download_se_eval_tool.sh
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
    local/wsj_format_data.sh

    tasks="tr_simu_${nch_train}ch dt_simu_${nch_train}ch dt_real_${nch_train}ch"
    for setname in dt_simu_8ch dt_real_8ch et_simu_8ch et_real_8ch; do
        echo ${setname}
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/utt2spk
        <data/${setname}/text sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        for ch in {A..H}; do
            <data/${setname}/wav.scp grep "_${ch}_" | sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
    done
    if [ ${nch_train} -eq 2 ]; then
        for setname in ${tasks}; do
            echo ${setname}
            mkdir -p data/${setname}_multich
            <data/${setname}/utt2spk sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/utt2spk
            <data/${setname}/text sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
            <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

            for ch in {A..B}; do
                <data/${setname}/wav.scp grep "_${ch}_" | sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
            done
            mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
        done
    elif [ ${nch_train} -ge 3 ] && [ ${nch_train} -le 8 ]; then
        ch_id_asc=$((64+nch_train))
        ch_id=$(awk 'BEGIN{printf "%c", '${ch_id_asc}'}')
        for setname in ${tasks}; do
            echo ${setname}
            mkdir -p data/${setname}_multich
            datasrc=$(echo ${setname} | sed 's/'${nch_train}'ch/8ch/g')
            <data/${datasrc}/utt2spk sed -r 's/^(.*?)_[A-'${ch_id}'](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/utt2spk
            <data/${datasrc}/text sed -r 's/^(.*?)_[A-'${ch_id}'](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
            <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

            for ch in $(eval echo "{A..${ch_id}}"); do
                <data/${datasrc}/wav.scp grep "_${ch}_" | sed -r 's/^(.*?)_[A-'${ch_id}'](_.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
            done
            mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
        done
    else
        log "Number of channel must between 2 and 8!"
        exit 1
    fi
fi

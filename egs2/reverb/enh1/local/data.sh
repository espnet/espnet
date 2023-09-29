#!/bin/bash

# Copyright 2018  Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Copyright 2023  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=2
nch_train=8 # number of training channels
compute_se=true # flag for turing on computation of dereverberation measures
enable_pesq=false # please make sure that you or your institution have the license to report PESQ before turning on this flag

help_message=$(cat << EOF
Usage: $0 [--nch_train <1-8>] [--compute_se <true/false>] [--enable_pesq <true/false>] [--stage <1-2>] [--stop_stage <1-2>]
EOF
)


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 0 ]; then
    log "${help_message}"
    exit 2
fi

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

    # (1) Download 327 MB data to data/local/reverb_tools
    # (2) Generate 7861 WAV files in ${REVERB_OUT}/WSJCAM0/data/primary_microphone/si_tr for simulation (1.8 GB)
    # (3) Simulate 62888 WAV files (training data) in ${REVERB_OUT}/REVERB_WSJCAM0_tr/data/mc_train/primary_microphone/si_tr
    #     via MATLAB (15 GB)
    local/generate_data.sh --wavdir ${REVERB_OUT} ${WSJCAM0}
    local/prepare_simu_data.sh --wavdir ${REVERB_OUT} ${REVERB} ${WSJCAM0}
    local/prepare_real_data.sh --wavdir ${REVERB_OUT} ${REVERB}

    # Run WPE and Beamformit
    # Generated WAV files are in ${REVERB_OUT}/WPE.
    local/run_wpe.sh
    local/run_beamform.sh ${REVERB_OUT}/WPE/

    if "${compute_se}"; then
        if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
            # download and install speech enhancement evaluation tools
            local/download_se_eval_tool.sh
        fi
        pesqdir="${PWD}/local"
        local/compute_se_scores.sh --nch "${nch_train}" --enable_pesq "${enable_pesq}" "${REVERB}" "${REVERB_OUT}" "${pesqdir}"
        cat "exp/compute_se_${nch_train}ch/scores/score_SimData"
        cat "exp/compute_se_${nch_train}ch/scores/score_RealData"
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
    for setname in dt_simu_8ch et_simu_8ch; do
        <data/${setname}_multich/wav.scp sed \
            -e 's#sox -M \(\S\+\)_ch[0-9].wav .*#\1.wav#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_dt/data/far_test/#/REVERB/REVERB_WSJCAM0_dt/data/cln_test/#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_dt/data/near_test/#/REVERB/REVERB_WSJCAM0_dt/data/cln_test/#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_et/data/far_test/#/REVERB/REVERB_WSJCAM0_et/data/cln_test/#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_et/data/near_test/#/REVERB/REVERB_WSJCAM0_et/data/cln_test/#g' \
            > data/${setname}_multich/spk1.scp

        # pad cln_test audios to the same length as near_test / far_test audios
        python local/pad_reverb_audios.py \
            data/${setname}_multich/spk1.scp \
            data/${setname}_multich/wav.scp \
            --audio_format .wav \
            --outfile data/${setname}_multich/spk1.scp
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

    <data/tr_simu_8ch_multich/wav.scp sed \
        -e 's#/REVERB/REVERB_WSJCAM0_tr/data/mc_train/#/REVERB/REVERB_WSJCAM0_tr/data/mc_train/clean_early/#g' \
        > data/tr_simu_8ch_multich/spk1.scp
    if [ ${nch_train} -ne 8 ]; then
        <data/tr_simu_${nch_train}ch_multich/wav.scp sed \
            -e 's#sox -M \(\S\+\)_ch[0-9].wav .*#\1.wav#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_tr/data/mc_train/#/REVERB/REVERB_WSJCAM0_tr/data/mc_train/clean_early/#g' \
            > data/tr_simu_${nch_train}ch_multich/spk1.scp

        <data/dt_simu_${nch_train}ch_multich/wav.scp sed \
            -e 's#sox -M \(\S\+\)_ch[0-9].wav .*#\1.wav#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_dt/data/far_test/#/REVERB/REVERB_WSJCAM0_dt/data/cln_test/#g' \
            -e 's#/REVERB/REVERB_WSJCAM0_dt/data/near_test/#/REVERB/REVERB_WSJCAM0_dt/data/cln_test/#g' \
            > data/dt_simu_${nch_train}ch_multich/spk1.scp

        # pad cln_test audios to the same length as near_test / far_test audios
        python local/pad_reverb_audios.py \
            data/dt_simu_${nch_train}ch_multich/spk1.scp \
            data/dt_simu_${nch_train}ch_multich/wav.scp \
            --audio_format .wav \
            --outfile data/dt_simu_${nch_train}ch_multich/spk1.scp
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

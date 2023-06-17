#!/usr/bin/env bash

# Copyright 2022  Carnegie Mellon University (Authors: Xuankai Chang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: max (Default), min
    [--sample_rate]: 16k (Default), 8k
EOF
)

# Path to the directory containing WHAM! noise
# (will download from the official site if not specified)
wham_noise=

stage=1
stop_stage=100
min_or_max=max
sample_rate=16k

librispeech_data_url=www.openslr.org/resources/12

log "$0 $*"
. utils/parse_options.sh

. ./db.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

cdir=$PWD


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Generating LibriMix to ./data/LibriMix"

    git clone https://github.com/JorisCos/LibriMix ./data/LibriMix

    # Download WHAM noise data
    if [ -z "${wham_noise}" ]; then
        # 17.65 GB unzipping to 35 GB
        mkdir -p ${cdir}/data/wham_noise
        wham_noise_url=https://storage.googleapis.com/whisper-public/wham_noise.zip
        wget --continue -O "${cdir}/data/wham_noise.zip" ${wham_noise_url}
        num_wavs=$(find "${cdir}/data/wham_noise" -iname "*.wav" | wc -l)
        if [ "${num_wavs}" = "4" ]; then
            echo "'${cdir}/data/wham_noise/' already exists. Skipping..."
        else
            unzip "${cdir}/data/wham_noise.zip" -d "${cdir}/data/"
        fi
        wham_noise="${cdir}/data/wham_noise"
    else
        # The simulation program will write data to wham_noie,
        # so copy it to user directory in case of permission issues.
        rsync -r -P "${wham_noise}" "${cdir}/data/wham_noise"
    fi

    (
    cd ./data/LibriMix
    librimix_outdir=./libri_mix_single


    python scripts/augment_train_noise.py --wham_dir ${cdir}/data/wham_noise
    # shellcheck disable=SC2043
    for n_src in 2;
    do
        metadata_dir=metadata/Libri$n_src"Mix"
        python scripts/create_librimix_from_metadata.py --librispeech_dir $LIBRISPEECH/LibriSpeech \
            --wham_dir ${cdir}/data/wham_noise \
            --metadata_dir $metadata_dir \
            --librimix_outdir $librimix_outdir \
            --n_src $n_src \
            --freqs ${sample_rate} \
            --modes ${min_or_max} \
            --types mix_clean mix_both mix_single
     done
    )

fi


if [ ! -e "./data/LibriMix" ]; then
    log "./data/LibriMix does not exist, please run stage 1 correctly."
    exit 1
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Audio Data Preparation"
    mkdir -p data/dev
    mkdir -p data/test
    mkdir -p data/train

    metadata_dir="data/LibriMix/libri_mix_single/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata"

    for dset in dev test train; do
        if [ "${dset}" = "train" ]; then
            mix_f="train-*"
        else
            mix_f=${dset}
        fi

        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $1}' > data/${dset}/utt2spk
        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $1}' > data/${dset}/spk2utt
        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $2}' > data/${dset}/wav.scp
        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $3}' > data/${dset}/spk1.scp
        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $4}' > data/${dset}/spk2.scp
        grep -v mixture_ID  ${metadata_dir}/mixture_${mix_f}_mix_both.csv | \
            sort -u | awk -F',' '{print $1, $5}' > data/${dset}/noise1.scp

    done

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LibriSpeech (dev / test / train_clean_*) Data Download."

    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
        echo "Data Download to ${LIBRISPEECH}"
        for part in dev-clean test-clean train-clean-100 train-clean-360; do
            local/download_and_untar.sh ${LIBRISPEECH} ${librispeech_data_url} ${part}
        done
    else
        log "stage 3: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: LibriSpeech Data Preparation"
    for part in dev-clean test-clean train-clean-100 train-clean-360; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} data/Librispeech/${part//-/_}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/Librispeech/train_clean_460 data/Librispeech/train_clean_100 data/Librispeech/train_clean_360

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: LibriMix Transcripts Files Prepration."

    librispeech_data_dir="data/Librispeech"

    for dset in dev test train; do
        if [ "${dset}" = "train" ]; then
            src_ls_dset="train_clean_460"
        else
            src_ls_dset=${dset}"_clean"
        fi

        for i in $(seq 2); do
            awk -v idx=${i} '(FILENAME==ARGV[1]) {text[$1]=$0} (FILENAME==ARGV[2]) {split($1, lst, "_"); uttname=lst[idx]; print($1, text[uttname])}' \
                ${librispeech_data_dir}/${src_ls_dset}/text data/${dset}/wav.scp | \
                cut -d" " -f1,3- > data/${dset}/text_spk${i}
        done

    done

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # use external data
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
        log "stage 6: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/
    fi

    if [ ! -e data/local/other_text/text ]; then
        # provide utterance id to each texts
        # e.g., librispeech_lng_00003686 A BANK CHECK
        zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
            awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) }' > data/local/other_text/text
    fi

fi


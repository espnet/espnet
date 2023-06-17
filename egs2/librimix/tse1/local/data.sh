#!/usr/bin/env bash

# Copyright 2022  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
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
Usage: $0
  optional argument:
    [--min_or_max]: min (Default), default is max
    [--sample_rate]: 8k (Default), default is 16k
    [--num_spk]: number of speakers in each mixture sample, default is 2
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


num_spk=2
min_or_max=max
sample_rate=16k

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" != "16k" ]] && [[ "$sample_rate" != "8k" ]]; then
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi
if [[ "$num_spk" != "2" ]] && [[ "$num_spk" != "3" ]]; then
  echo "Error: num_spk must be either 2 or 3"
  exit 1
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

librimix=data/LibriMix/libri_mix/Libri2Mix
mkdir -p data/{train,dev,test}
tmpdir=$(mktemp -d data/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: LibriMix Data Simulation"

    local/librimix_data.sh --min_or_max ${min_or_max} --sample_rate ${sample_rate} --num_spk ${num_spk}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"

    for f in "${librimix}/wav${sample_rate}/${min_or_max}/metadata"/mixture_train-*_mix_both.csv; do
        grep -v mixture_ID "$f"
    done | sort -u > "${tmpdir}/train.csv"
    grep -v mixture_ID "${librimix}"/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | sort -u > "${tmpdir}/dev.csv"
    grep -v mixture_ID "${librimix}"/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u > "${tmpdir}/test.csv"

    for dset in train dev test; do
        awk -F ',' '{print $1, $2}' "${tmpdir}/${dset}.csv" > data/${dset}/wav.scp
        awk -F ',' '{print $1, $1}' "${tmpdir}/${dset}.csv" > data/${dset}/utt2spk
        cp data/${dset}/utt2spk data/${dset}/spk2utt

        if [ "$num_spk" = "2" ]; then
            awk -F ',' '{print $1, $3}'  "${tmpdir}/${dset}.csv" > data/${dset}/spk1.scp
            awk -F ',' '{print $1, $4}'  "${tmpdir}/${dset}.csv" > data/${dset}/spk2.scp
            awk -F ',' '{print $1, $5}'  "${tmpdir}/${dset}.csv" > data/${dset}/noise1.scp
        elif [ "$num_spk" = "3" ]; then
            awk -F ',' '{print $1, $3}'  "${tmpdir}/${dset}.csv" > data/${dset}/spk1.scp
            awk -F ',' '{print $1, $4}'  "${tmpdir}/${dset}.csv" > data/${dset}/spk2.scp
            awk -F ',' '{print $1, $5}'  "${tmpdir}/${dset}.csv" > data/${dset}/spk3.scp
            awk -F ',' '{print $1, $6}'  "${tmpdir}/${dset}.csv" > data/${dset}/noise1.scp
        fi
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare LibriMix target-speaker enroll signal"

    python local/prepare_spk2enroll_librispeech.py \
        "${librimix}/wav${sample_rate}/${min_or_max}/train-100" \
        "${librimix}/wav${sample_rate}/${min_or_max}/train-360" \
        --is_librimix True \
        --outfile data/train/spk2enroll.json \
        --audio_format wav

    python local/prepare_spk2enroll_librispeech.py \
        "${librimix}/wav${sample_rate}/${min_or_max}/dev" \
        --is_librimix True \
        --outfile data/dev/spk2enroll.json \
        --audio_format wav

    python local/prepare_spk2enroll_librispeech.py \
        "${librimix}/wav${sample_rate}/${min_or_max}/test" \
        --is_librimix True \
        --outfile data/test/spk2enroll.json \
        --audio_format wav

    if [ $num_spk -eq 2 ]; then
        wget -O "data/dev/mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment
        wget -O "data/test/mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment
    else
        wget -O "data/dev/mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri3mix/data/wav8k/min/dev/map_mixture2enrollment
        wget -O "data/test/mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri3mix/data/wav8k/min/test/map_mixture2enrollment
    fi

    for dset in train dev test; do
        if [ "${dset}" = "train" ]; then
            is_train=True
            # This script generates enroll_spk?.scp under "data/${dset}"
            python local/prepare_librimix_enroll.py \
                data/${dset}/wav.scp \
                data/${dset}/spk2enroll.json \
                --num_spk ${num_spk} \
                --train ${is_train} \
                --seed 1 \
                --output_dir data/${dset} \
                --outfile_prefix "enroll_spk"
        else
            is_train=False
            python local/prepare_librimix_enroll.py \
                data/${dset}/wav.scp \
                data/${dset}/spk2enroll.json \
                --librimix_dir "${librimix}/wav${sample_rate}/${min_or_max}" \
                --mix2enroll "data/${dset}/mixture2enrollment" \
                --num_spk ${num_spk} \
                --train ${is_train} \
                --output_dir data/${dset} \
                --outfile_prefix "enroll_spk"
        fi
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare data files for train-100 and train-360"
    mkdir -p data/{train-100,train-360}

    for subset in "train-100" "train-360"; do
        grep -e "${subset}" "data/train/wav.scp" > "data/${subset}/wav.scp"
        for f in data/train/*.scp; do
            [ "$f" = "data/train/wav.scp" ] || utils/filter_scp.pl "data/${subset}/wav.scp" "$f" > "data/${subset}/$(basename $f)"
        done
        utils/filter_scp.pl "data/${subset}/wav.scp" data/train/utt2spk > data/${subset}/utt2spk

        ln -s ../../data/train/spk2enroll.json data/${subset}/spk2enroll.json
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

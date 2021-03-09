#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

# Path to the directory containing WHAM! noise
# (will download from the official site if not specified)
wham_noise=

min_or_max=max
sample_rate=16k

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

cdir=$PWD


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
  python scripts/create_librimix_from_metadata.py --librispeech_dir $LIBRISPEECH \
    --wham_dir ${cdir}/data/wham_noise \
    --metadata_dir $metadata_dir \
    --librimix_outdir $librimix_outdir \
    --n_src $n_src \
    --freqs 8k 16k \
    --modes min max \
    --types mix_clean mix_both mix_single
  done
)

mkdir -p data/dev
mkdir -p data/test
mkdir -p data/train
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | sort -u | awk -F ',' '{print $1, $2}' > data/dev/wav.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | grep -v mixture_ID | sort -u | awk -F ',' '{print $1, $3}' > data/dev/spk1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | grep -v mixture_ID | sort -u | awk -F ',' '{print $1, $4}' > data/dev/spk2.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | grep -v mixture_ID | sort -u | awk -F ',' '{print $1, $5}' > data/dev/noise1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | grep -v mixture_ID | sort -u | awk -F ',' '{print $1, $1}' > data/dev/utt2spk
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_dev_mix_both.csv | grep -v mixture_ID | sort -u | awk -F ',' '{print $1, $1}' > data/dev/spk2utt

grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $2}' > data/test/wav.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $3}' > data/test/spk1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $4}' > data/test/spk2.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $5}' > data/test/noise1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $1}' > data/test/utt2spk
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_test_mix_both.csv | sort -u | awk -F ',' '{print $1, $1}' > data/test/spk2utt

grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $2}' > data/train/wav.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $3}' > data/train/spk1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $4}' > data/train/spk2.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $5}' > data/train/noise1.scp
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $1}' > data/train/utt2spk
grep -v mixture_ID data/LibriMix/libri_mix/Libri2Mix/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | sort -u | awk -F ',' '{print $1, $1}' > data/train/spk2utt

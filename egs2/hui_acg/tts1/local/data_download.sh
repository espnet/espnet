#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

download_dir=$1
spk=$2

available_spks=(
    "Bernd_Ungerer" "Hokuspokus" "Friedrich" "Karlsson" "Eva_K" "awb" "ksp"
)
base_url=https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <spk>"
    exit 1
fi

# check spk
if ! echo "${available_spks[*]}" | grep -q "${spk}"; then
    echo "Specified spk (${spk}) is not available or not supported." >&2
    echo "Available spk: ${available_spks[*]}" >&2
    exit 1
fi

set -euo pipefail

if [ ! -e "${download_dir}/${spk}" ]; then
    mkdir -p "${download_dir}"
    wget "${base_url}/${spk}_Clean.zip" -P "${download_dir}"
    unzip "${download_dir}/${spk}_Clean.zip" -d "${download_dir}"
    echo "Successfully finished download and unzip."
else
    echo "Already exists. Skipped."
fi

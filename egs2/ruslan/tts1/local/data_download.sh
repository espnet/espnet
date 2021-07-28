#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

if [ ! -e "${download_dir}/RUSLAN" ]; then
    (
        mkdir -p "${download_dir}"
        cd "${download_dir}"
        gdown "https://drive.google.com/uc?id=1Y6vv--gcDx-S8DieSGaD7WnB86kZLgc_"
        jar xf RUSLAN.zip
    )
    echo "Successfully finished download and unzip wav files."
else
    echo "Already exists. Skipped."
fi
if [ ! -e "${download_dir}/RUSLAN/metadata.csv" ]; then
    (
        mkdir -p "${download_dir}"
        cd "${download_dir}"
        gdown "https://drive.google.com/uc?id=11TD_ZwIOo-Wo75GYv-OWWOS3ABmwmAdK"
        mv -v metadata_RUSLAN_22200.csv RUSLAN/metadata.csv
    )
    echo "Successfully finished download metadata."
else
    echo "Already exists. Skipped."
fi

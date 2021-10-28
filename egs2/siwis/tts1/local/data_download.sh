#!/usr/bin/env bash

# Copyright 2021 Takenori Yoshimura
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

if [ ! -e "${download_dir}/SiwisFrenchSpeechSynthesisDatabase" ]; then
    (
        mkdir -p "${download_dir}"
        cd "${download_dir}"
        wget https://datashare.ed.ac.uk/download/DS_10283_2353.zip --no-check-certificate
        unzip DS_10283_2353.zip
        unzip -q SiwisFrenchSpeechSynthesisDatabase.zip
    )
    echo "Successfully finished download and unzip wav files."
else
    echo "Already exists. Skipped."
fi

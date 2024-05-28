#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${db_root}/hi_fi_tts_v0" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    if ! [ -f "./hi_fi_tts_v0.tar.gz" ]; then
        echo "Downloading HiFi-TTS dataset..."
        wget https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz
    else
        echo "HiFi-TTS dataset zip already exists. Skipped."
    fi
    tar xvzf ./hi_fi_tts_v0.tar.gz
    rm ./hi_fi_tts_v0.tar.gz
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

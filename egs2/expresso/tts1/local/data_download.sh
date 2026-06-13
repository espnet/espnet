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
if [ ! -e "${db_root}" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    if ! [ -f "./expresso.tar" ]; then
        echo "Downloading EXPRESSO dataset..."
        wget https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar
    else
        echo "EXPRESSO dataset tar already exists. Skipped."
    fi
    tar xvf ./expresso.tar
    rm ./expresso.tar
    rm -rf "expresso/audio_48khz/conversational"
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

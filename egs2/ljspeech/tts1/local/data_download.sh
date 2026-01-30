#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/LJSpeech-1.1" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"
    wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -vxf ./*.tar.bz2
    rm ./*.tar.bz2
    cd "${cwd}"
    echo "successfully prepared data."
else
    echo "already exists. skipped."
fi

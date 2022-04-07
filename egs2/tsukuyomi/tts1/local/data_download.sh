#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

if [ ! -e "${download_dir}/tsukuyomi_chan_corpus" ]; then
    (
        mkdir -p "${download_dir}"
        cd "${download_dir}"
        wget "https://tyc.rei-yumesaki.net/files/sozai-tyc-corpus1.zip"
        LC_ALL="" unzip -O sjis ./sozai-tyc-corpus1.zip
        LC_ALL="" mv つくよみちゃん* tsukuyomi_chan_corpus
    )
    echo "Successfully finished download and unzip wav files."
else
    echo "Already exists. Skipped."
fi

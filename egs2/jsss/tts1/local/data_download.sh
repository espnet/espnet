#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Download JSSS Corpus

. ./path.sh || exit 1

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

url="https://drive.google.com/a/g.sp.m.is.nagoya-u.ac.jp/uc?id=1NyiZCXkYTdYBNtD1B-IMAYCVa-0SQsKX"
if [ ! -e "${download_dir}/jsss_ver1" ]; then
    scripts/utils/download_from_google_drive.sh \
        "${url}" "${download_dir}" zip
    echo "Successfully downloaded JSSS corpus."
else
    echo "Already exists. Skipped."
fi

cwd=$(pwd)
if [ ! -e "${download_dir}/JSSSLabel" ]; then
    echo "Downloading phoneme labels for jsss_ver1"
    cd "${download_dir}"
    git clone https://github.com/kan-bayashi/JSSSLabel
    for name in long-form short-form simplification summarization; do
        cp -vr JSSSLabel/${name} jsss_ver1
    done
    cd "${cwd}"
    echo "Successfully downloaded JSSS label."
else
    echo "Already exists. Skipped."
fi

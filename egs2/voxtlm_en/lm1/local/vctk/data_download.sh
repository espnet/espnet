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
if [ ! -e "${db_root}/VCTK-Corpus" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz
    tar xvzf ./VCTK-Corpus.tar.gz
    rm ./VCTK-Corpus.tar.gz
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

if [ ! -e "${db_root}/VCTK-Corpus/lab" ]; then
    cd "${db_root}" || exit 1;
    git clone https://github.com/kan-bayashi/VCTKCorpusFullContextLabel.git
    cp -r VCTKCorpusFullContextLabel/lab ./VCTK-Corpus
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded label data."
else
    echo "Already exists. Skipped."
fi

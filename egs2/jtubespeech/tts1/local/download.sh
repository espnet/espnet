#!/usr/bin/env bash

# Copyright 2021 Takaaki Saeki (The University of Tokyo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e ${download_dir}/jtuberaw ]; then
    mkdir -p ${download_dir}
    cd ${download_dir}
    FILE_NAME=jtuberaw.tar.gz
    gdown "https://drive.google.com/uc?id=1X_harC0e1tjMX1FtCldD67XOysQuq_Ib"
    tar -zxvf ${FILE_NAME} jtuberaw
    rm -rf ${FILE_NAME}
    cd ${cwd}
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

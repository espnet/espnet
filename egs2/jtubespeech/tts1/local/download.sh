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

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

cwd=$(pwd)
if [ ! -e ${download_dir}/jtuberaw ]; then
    mkdir -p ${download_dir}
    cd ${download_dir}
    FILE_NAME=jtubetest.tar.gz
    gdrive_download 1Ogk7NlLfqzYYFyhjXVmIecnIdEmy_t1g ${FILE_NAME}
    tar -zxvf ${FILE_NAME} jtuberaw
    rm -rf ${FILE_NAME}
    cd ${cwd}
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi
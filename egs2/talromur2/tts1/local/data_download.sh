#!/usr/bin/env bash

# Copyright 2022 Gunnar Thor Örnólfsson
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e ${download_dir} ]; then
    mkdir -p "${download_dir}"
fi
cd "${download_dir}"
if [ ! -e "asdfasdf" ]; then
    echo "fetching the Talrómur 2 dataset"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/167/talromur2.zip
else
    echo "Data already present. Skipping..."
fi
unzip *.zip
rm ./*.zip
cd "${cwd}"
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
if [ ! -e "f" ]; then
    echo "fetching speaker f"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/alfur.zip
else
    echo "f already exists, skipping..."
fi
if [ ! -e "b" ]; then
    echo "fetching speaker b"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/bjartur.zip
else
    echo "b already exists, skipping..."
fi
if [ ! -e "d" ]; then
    echo "fetching speaker d"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/bui.zip
else
    echo "d already exists, skipping..."
fi
if [ ! -e "c" ]; then
    echo "fetching speaker c"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/dilja.zip
else
    echo "c already exists, skipping..."
fi
if [ ! -e "a" ]; then
    echo "fetching speaker a"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/rosa.zip
else
    echo "a already exists, skipping..."
fi
if [ ! -e "h" ]; then
    echo "fetching speaker h"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/steinn.zip
else
    echo "h already exists, skipping..."
fi
if [ ! -e "g" ]; then
    echo "fetching speaker g"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/salka.zip
else
    echo "g already exists, skipping..."
fi
if [ ! -e "e" ]; then
    echo "fetching speaker e"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/104/ugla.zip
else
    echo "e already exists, skipping..."
fi
unzip *.zip
rm ./*.zip
cd "${cwd}"
echo "successfully fetched data."
if [ ! -e "${download_dir}/split"]; then
    cd "${download_dir}"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/talromur-splits.zip
    unzip *.zip
    rm ./*.zip
    cd "${cwd}"
    echo "successfully fetched splits."
else
    echo "splits already present. skipped."
fi
if [ ! -e "${download_dir}/alignments"]; then
    mkdir -p "${download_dir}/alignments"
    cd "${download_dir}/alignments"
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/a.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/b.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/c.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/d.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/e.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/f.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/g.zip
    wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/201/h.zip
    unzip *.zip
    rm ./*.zip
    cd "${cwd}"
    echo "successfully fetched alignments."
else
    echo "alignments already present. skipped."
fi
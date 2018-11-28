#!/bin/bash

# Copyright 2012 Vassil Panayotov
#           2017 Johns Hopkins University (author: Shinji Watanabe)
# Apache 2.0

# Downloads and extracts multilingual data from VoxForge website

. ./path.sh

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "usage: $0 <lang> <datadir>"
    exit 1;
fi

lang=$1
DATA_ROOT=$2/${lang}
if   [ ${lang} = 'en' ]; then
    DATA_SRC="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit" 
elif [ ${lang} = 'nl' ]; then
    DATA_SRC="http://www.repository.voxforge1.org/downloads/Dutch/Trunk/Audio/Main/16kHz_16bit" 
elif [ ${lang} = 'ru' ]; then
    DATA_SRC="http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit" 
else
    DATA_SRC="http://www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit" 
fi
DATA_TGZ=${DATA_ROOT}/tgz
DATA_EXTRACT=${DATA_ROOT}/extracted

mkdir -p ${DATA_TGZ} 2>/dev/null

# Check if the executables needed for this script are present in the system
command -v wget >/dev/null 2>&1 ||\
    { echo "\"wget\" is needed but not found"'!'; exit 1; }

echo "--- Starting VoxForge data download (may take some time) ..."
wget -P ${DATA_TGZ} -l 1 -N -nd -c -e robots=off -A tgz -r -np ${DATA_SRC} || \
    { echo "WGET error"'!' ; exit 1 ; }

mkdir -p ${DATA_EXTRACT}

echo "--- Starting VoxForge archives extraction ..."
for a in ${DATA_TGZ}/*.tgz; do
    tar -C ${DATA_EXTRACT} -xf ${a}
done

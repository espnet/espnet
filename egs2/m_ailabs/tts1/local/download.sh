#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
lang=$2

available_langs=(
    "de_DE" "en_UK" "it_IT" "es_ES" "en_US" "fr_FR" "uk_UK" "ru_RU" "pl_PL"
)

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <tar_dir> <lang_tag>"
    echo "Available languages: ${available_langs[*]}"
    exit 1
fi

# check language
if ! $(echo ${available_langs[*]} | grep -q ${lang}); then
    echo "Specified language is not available or not supported."
    exit 1
fi

# download dataset
cwd=`pwd`
if [ ! -e ${db}/${lang} ]; then
    mkdir -p ${db}
    cd ${db}
    wget http://www.caito.de/data/Training/stt_tts/${lang}.tgz
    tar xvf ${lang}.tgz
    rm ${lang}.tgz
    cd $cwd
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi

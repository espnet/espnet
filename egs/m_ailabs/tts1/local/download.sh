#!/bin/bash -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
lang=$2

available_langs=(
    "de_DE" "en_UK" "it_IT" "es_ES" "en_US" "fr_FR" "uk_UK" "ru_RU"
)

# check arguments
<<<<<<< HEAD
if [ $# != 2 ]; then
=======
if [ $# != 2 ];then
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    echo "Usage: $0 <tar_dir> <lang_tag>"
    echo "Available languages: ${available_langs[*]}"
    exit 1
fi

# check language
<<<<<<< HEAD
if ! $(echo ${available_langs[*]} | grep -q ${lang}); then
=======
if ! $(echo ${available_langs[*]} | grep -q ${lang});then
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    echo "Specified language is not available or not supported."
    exit 1
fi

# download dataset
cwd=`pwd`
<<<<<<< HEAD
if [ ! -e ${db}/${lang} ]; then
=======
if [ ! -e ${db}/${lang} ];then
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
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

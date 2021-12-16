#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
id=$2
pw=$3

cwd=`pwd`
if [ ! -e ${db}/LJSpeech-1.1 ]; then
    mkdir -p ${db}
    cd ${db}
    wget --http-user=${id} --http-password=${pw} http://data.cstr.ed.ac.uk/blizzard2017-18/usborne/2018/2018_EH1/blizzard_release_2017_v2.zip
    unzip ./*.zip
    rm ./*.zip
    cd $cwd
fi

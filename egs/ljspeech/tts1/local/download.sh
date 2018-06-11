#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

cwd=`pwd`
if [ ! -e ${db}/LJSpeech-1.1 ];then
    mkdir -p ${db}
    cd ${db}
    wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -vxf ./*.tar.bz2
    rm ./*.tar.bz2
    cd $cwd
fi

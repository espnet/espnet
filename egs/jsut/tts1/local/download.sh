#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

cwd=`pwd`
if [ ! -e ${db}/jsut_ver1.1 ];then
    mkdir -p ${db}
    cd ${db}
    wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o ./*.zip
    rm ./*.zip
    cd $cwd
fi

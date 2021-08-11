#!/usr/bin/env bash
# Copyright 2021    Indian Institute of Science (author: Sathvik Udupa)
# Apache 2.0

available_languages=(
    "hi-en" "bn-en"
)
db=$1
lang=$2

if [ $# != 2 ]; then
    echo "Usage: $0 <db_root_dir> <spk>"
    echo "Available langauges for mucs subtask2: ${available_languages[*]}"
    exit 1
fi

if ! $(echo ${available_languages[*]} | grep -q ${lang}); then
    echo "Specified langauge (${lang}) is not available or not supported." >&2
    echo "Choose from: ${available_languages[*]}"
    exit 1
fi

declare -A trainset
trainset['hi-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi-English_train.tar.gz'
trainset['bn-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Bengali-English_train.tar.gz'

declare -A valset
valset['hi-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi-English_test.tar.gz'
valset['bn-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Bengali-English_test.tar.gz'

declare -A testset
testset['hi-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/subtask2_blindtest_wReadme.tar.gz'
testset['bn-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/subtask2_blindtest_wReadme.tar.gz'

cwd=`pwd`
if [ ! -e ${db}/${lang}.done ]; then
    mkdir -p ${db}
    cd ${db}
    mkdir -p ${lang}
    cd ${lang}
    wget -O valid.zip ${valset[$lang]}
    tar xf "valid.zip"
    rm valid.zip
    wget -O test.zip ${testset[$lang]}
    tar xf "test.zip"
    rm test.zip
    wget -O train.zip ${trainset[$lang]}
    tar xf "train.zip"
    rm train.zip
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/${lang}.done
else
    echo "Already exists. Skip download."
fi

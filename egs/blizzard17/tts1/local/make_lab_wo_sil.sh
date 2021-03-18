#!/usr/bin/env bash

# Copyright 2018 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

TMP=./local/tmp
LAB=lab
mkdir -p ${TMP}

set -euo pipefail

find $1/fls/*/${LAB} -name "*.lab" | xargs dirname | xargs dirname | sort | uniq > ${TMP}/dir_list.txt
cat ${TMP}/dir_list.txt | while read -r dir;do

    echo ${dir}

    for ftype in "m4a" "mp3" "wma";do
        find ${dir}/audio/ -name "*.${ftype}" | sort | while read -r audio_file;do
            lab_file=$(echo ${audio_file} | sed -e "s/audio/${LAB}/g" -e "s/${ftype}/lab/g" )
            lab_wo_sil_file=$(echo ${audio_file} | sed -e "s/audio/${LAB}_wo_sil/g" -e "s/${ftype}/lab/g")
            echo ${lab_wo_sil_file} | xargs dirname | xargs mkdir -p
            python3 local/make_lab_wo_sil.py ${audio_file} ${lab_file} ${lab_wo_sil_file}
        done
    done

done

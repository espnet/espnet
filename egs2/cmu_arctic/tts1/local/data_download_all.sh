#!/usr/bin/env bash

# Copyright 2021 Peter Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
# download all speakers
for spk in slt clb bdl rms jmk awb ksp
do
    if [ ! -e "${db_root}/${spk}.done" ]; then
        
        mkdir -p "${db_root}"
        cd "${db_root}" || exit 1;
        wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_${spk}_arctic-0.95-release.tar.bz2
        tar xf cmu_us_${spk}*.tar.bz2
        rm cmu_us_${spk}*.tar.bz2
        cd "${cwd}" || exit 1;
        echo "Speaker" $spk ": Successfully finished download."
        touch ${db_root}/${spk}.done
      else
        echo "Speaker" $spk ": Already exists. Skip download."
    fi
done

# combine the data
mkdir -p ${db_root}/cmu_us_all_arctic/wav
mkdir -p ${db_root}/cmu_us_all_arctic/lab
mkdir -p ${db_root}/cmu_us_all_arctic/etc

if [ ! -e "${db_root}/cmu_us_all_arctic/txt.done.data" ]; then
    cd . > ${db_root}/cmu_us_all_arctic/etc/txt.done.data
    for spk in slt clb bdl rms jmk awb ksp
    do
        echo "Combine Speaker: "$spk
        if [ $(ls ${db_root}/cmu_us_all_arctic/wav/${spk}* | wc -l) -lt 1132 ]; then
            find ${db_root}/cmu_us_${spk}_arctic/wav -name "arctic*.wav" -follow | sort | while read -r filename;do
                cp ${filename} ${db_root}/cmu_us_all_arctic/wav/${spk}_$(basename $filename)
            done
        fi

        if [ $(ls ${db_root}/cmu_us_all_arctic/lab/${spk}* | wc -l) -lt 1132 ]; then
            find ${db_root}/cmu_us_${spk}_arctic/lab -name "arctic*.lab" -follow | sort | while read -r filename;do
                cp ${filename} ${db_root}/cmu_us_all_arctic/lab/${spk}_$(basename $filename)
            done
        fi
        sed  's/arctic_/'"${spk}"'_arctic_/' ${db_root}/cmu_us_${spk}_arctic/etc/txt.done.data >> ${db_root}/cmu_us_all_arctic/etc/txt.done.data
    done
fi

file_num_wav=$(ls ${db_root}/cmu_us_all_arctic/wav | wc -l)
file_num_lab=$(ls ${db_root}/cmu_us_all_arctic/lab | wc -l)
if [ $file_num_wav -ne $file_num_lab ] ; then
    echo "Error: Wrong file number: " $file_num ". Some files are missing."
    exit 1
fi

if [ $file_num_wav -lt 7924 ] ; then
    echo "Error: Less audio files found: " $file_num ", expected 7924. Some files are missing."
    exit 1
fi

echo "Successfully finished download and combine all speakers."

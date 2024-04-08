#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

srcspks=(
    "SEF1" "SEF2" "SEM1" "SEM2"
)

trgspks_task1=(
    "TEF1" "TEF2" "TEM1" "TEM2"
)

trgspks_task2=(
    "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
)

# download dataset
cwd=`pwd`
if [ ! -e ${db}/.done ]; then
    mkdir -p ${db}
    cd ${db}
    git clone https://github.com/nii-yamagishilab/VCC2020-database.git

    cd VCC2020-database
    unzip '*.zip'
    rm -rf __MACOSX/ # remove extra folder

    # integrate source waveforms
    for srcspk in "${srcspks[@]}"; do
        mv vcc2020_database_evaluation/${srcspk}/*.wav source/${srcspk}/
    done
    mv source/* ./

    # integrate target waveforms
    for trgspk in "${trgspks_task1[@]}"; do
        mv vcc2020_database_groundtruth/${trgspk}/*.wav target_task1/${trgspk}/
    done
    for trgspk in "${trgspks_task2[@]}"; do
        mv vcc2020_database_groundtruth/${trgspk}/*.wav target_task2/${trgspk}/
    done
    mv target_task1/* ./
    mv target_task2/* ./

    # move transcriptions
    mkdir prompts
    cat vcc2020_database_transcriptions/transcriptions_training/vcc2020_database_training_Eng_transcriptions.txt \
        vcc2020_database_transcriptions/transcriptions_evaluation/vcc2020_database_evaluation_transcriptions.txt > prompts/Eng_transcriptions.txt
    mv vcc2020_database_transcriptions/transcriptions_training/vcc2020_database_training_Fin_transcriptions.txt prompts/Fin_transcriptions.txt
    mv vcc2020_database_transcriptions/transcriptions_training/vcc2020_database_training_Ger_transcriptions.txt prompts/Ger_transcriptions.txt
    mv vcc2020_database_transcriptions/transcriptions_training/vcc2020_database_training_Man_transcriptions.txt prompts/Man_transcriptions.txt
    rm -rf vcc2020_database_transcriptions

    # delete folders and files
    rm -f *.zip
    rm -rf source
    rm -rf target_task1
    rm -rf target_task2
    rm -rf vcc2020_database_groundtruth/
    rm -rf vcc2020_database_evaluation/
    cd ..
    mv VCC2020-database/* ./
    rm -rf VCC2020-database
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/.done
else
    echo "Already exists. Skip download."
fi

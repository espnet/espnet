#!/bin/bash

# Copyright 2019 Nagoya University (Koichi Miyazaki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # pytorch only
stage=3         # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)

model=baseline


set -e
set -u


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    ### Note: It may not work on python3.7
    echo "stage 0: Data Preparation"
#    git clone https://github.com/turpaultn/DCASE2019_task4.git
    # TODO: run in shell script. fix relative path problem
    #    cd DCASE2019_task4/baseline
    #    python ./download_data.py
    #    patch label_index.patch
    wget https://zenodo.org/record/2583796/files/Synthetic_dataset.zip -O ./DCASE2019_task4/dataset/Synthetic_dataset.zip
    unzip ./DCASE2019_task4/dataset/Synthetic_dataset.zip -d ./DCASE2019_task4/dataset
    rm ./DCASE2019_task4/dataset/Synthetic_dataset.zip
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    python ./local/preprocess_data.py
    for x in train validation; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        . ./local/make_fbank.sh data/${x} exp/make_fbank/train fbank
    done
fi

# TODO: modify data loader
#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#    echo "stage 3: Network Training"
#    python ./DCASE2019_task4/baseline/main.py
#fi
#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (author: Shinji Watanabe)
#           2020 The University of Tokyo (author:Zixiong Su)
# Apache 2.0

# Downloads extracted features and text data from SSC ftp server.
# You can specify the feature directory on the server to try different features.
# Available directories:
# DCT_Features/features/{5,10,20,30}-dct
# Auto_Features/features/{5,10,20,30}d
# e.g., ssc_data_prepare.sh --feat_dir Auto_Features/features/30d
#       ssc_data_prepare.sh --feat_dir DCT_Features/features/20-dct"

. ./path.sh

feat_dir=$1

feat_local_dir=data/${feat_dir}
mkdir -p ${feat_local_dir}
wget -c -N ftp://ftp.espci.fr/pub/sigma/Features/${feat_dir}/* -P ${feat_local_dir}
wget -c -N https://ftp.espci.fr/pub/sigma/TIMIT_training/TIMIT_Transcripts.txt -P ${feat_local_dir}
wget -c -N https://ftp.espci.fr/pub/sigma/WSJ05K_Test/WSJ0_5K_Transcripts.txt -P ${feat_local_dir}

mkdir -p data/train
mkdir -p data/test

sed "s/^[0-9]*[.].//g" ${feat_local_dir}/TIMIT_Transcripts.txt  | sed "s/([^)]*)//g" | tr [:lower:] [:upper:] > data/train/text
sed "s/^[0-9]*[.].//g" ${feat_local_dir}/WSJ0_5K_Transcripts.txt | sed "s/([^)]*)//g" | tr [:lower:] [:upper:] > data/test/text
for x in train test; do
    sed "s% .*mfcc/% "${PWD}"/"${feat_local_dir}"/%g" ${feat_local_dir}/${x}*.scp \
    | sed "s/t\([0-9]\)_/t0\1_/g" | sort -o data/${x}/feats.scp
done

local/featprepare.py

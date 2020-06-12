#!/bin/bash

# Copyright 2017 Johns Hopkins University (author: Shinji Watanabe)
#           2020 The University of Tokyo (author:Zixiong Su)
# Apache 2.0

# Downloads extracted features and text data from SSC ftp server.
. ./path.sh

timit_dir=data/timit_data/
mkdir -p ${timit_dir} ${timit_data}
wget -c -N ftp://ftp.espci.fr/pub/sigma/Features/DCT_Features/features/30-dct/* -P ${timit_dir}
wget https://ftp.espci.fr/pub/sigma/TIMIT_training/TIMIT_Transcripts.txt -P ${timit_dir}
wget https://ftp.espci.fr/pub/sigma/WSJ05K_Test/WSJ0_5K_Transcripts.txt -P ${timit_dir}

sed "s/^[0-9]*[.].//g" ${timit_dir}/TIMIT_Transcripts.txt  | sed "s/([^)]*)//g" | tr [:lower:] [:upper:] > ${timit_dir}/train.txt
sed "s/^[0-9]*[.].//g" ${timit_dir}/WSJ0_5K_Transcripts.txt | sed "s/([^)]*)//g" | tr [:lower:] [:upper:] > ${timit_dir}/test.txt
for x in train test;do
    cat ${timit_dir}/${x}-feats.scp \
    | sed "s%/home/jiyan/kaldi-update/egs/jiyan/dct30-wsj/voxforge/mfcc/%"${PWD}"/data/timit_data/%g" > data/${x}/feats.scp
done

mkdir -p data/train
mkdir -p data/test
local/featprepare.py


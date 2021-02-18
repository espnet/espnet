#! /usr/bin/env bash
. ./path.sh || exit 1;
datadir=$1 
mkdir -p $datadir
pushd $datadir
wget -O val.csv.bz2 https://datasets.kensho.com/api/v1/download/val.csv.bz2/2ba05f3cc0e041d086a90dfd151d4c3c
bzip2 -d val.csv.bz2
wget -O train.csv.bz2 https://datasets.kensho.com/api/v1/download/train.csv.bz2/2ba05f3cc0e041d086a90dfd151d4c3c
bzip2 -d train.csv.bz2 
wget -O metadata.txt https://datasets.kensho.com/api/v1/download/metadata.txt/2ba05f3cc0e041d086a90dfd151d4c3c
#wget -O val.tar.gz https://datasets.kensho.com/api/v1/download/val.tar.bz2/2ba05f3cc0e041d086a90dfd151d4c3c
#tar -xvf val.tar.gz
#wget -O train.tar.gz https://applyyourdata
#tar -xvf train.tar.gz
popd

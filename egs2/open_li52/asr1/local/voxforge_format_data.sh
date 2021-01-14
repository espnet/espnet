#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0

source ./path.sh

echo "=== Preparing all data ..."

if [ $# -ne 1 ]; then
    echo "Usage: $0 <lang>"
    exit 1
fi

lang=$1
srcdir=data/local/$lang

for x in all; do
    mkdir -p data/${x}_$lang
    cp $srcdir/${x}_wav.scp data/${x}_${lang}/wav.scp || exit 1;
    cp $srcdir/${x}_trans.txt data/${x}_${lang}/text || exit 1;
    cp $srcdir/$x.spk2utt data/${x}_${lang}/spk2utt || exit 1;
    cp $srcdir/$x.utt2spk data/${x}_${lang}/utt2spk || exit 1;
done

echo "*** Succeeded in formatting data."

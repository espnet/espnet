#!/usr/bin/env bash
# Copyright 2021 Author: Yuekai Zhang
# Apache 2.0


. ./path.sh || exit 1;
src=$1
dst=$2
mkdir -p $dst

echo "train_split.sh: using first 10000 items as tr_dev"
num=10000

for f in wav.scp text spk2utt utt2spk; do
  N=$(wc -l < $src/$f)
  L=$(($N - $num))
  head -n $num $src/$f > $dst/$f
  mv $src/$f $src/bak$f
  tail -n $L   $src/bak$f > $src/$f
done
echo "local/train_split.sh succeeded"
exit 0;

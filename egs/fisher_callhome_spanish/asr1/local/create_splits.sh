#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

data_dir=data
train_all=data/train_all

if [ $# -lt 1 ]; then
    echo "Specify the location of the split files"
    exit 1;
fi

splitFile=$1

# Train first
for split in train dev test dev2
do

  cp -r $train_all $data_dir/$split

  awk 'BEGIN {FS=" "}; FNR==NR { a[$1]; next } ((substr($2,0,length($2)-2) ".sph") in a)' \
  $splitFile/$split $train_all/segments > $data_dir/$split/segments

  n=`awk 'BEGIN {FS = " "}; {print substr($2,0,length($2)-2)}' $data_dir/$split/segments | sort | uniq | wc -l`

  echo "$n conversations left in split $split"

  utils/fix_data_dir.sh $data_dir/$split
  utils/validate_data_dir.sh $data_dir/$split
done

#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

dataname= # Expects _name
#train_all=data/callhome_train_all

. ./utils/parse_options.sh

if [ $# -lt 1 ]; then
    echo "Specify the location of the split files"
    echo "and the names of the splits"
    exit 1;
fi

splitFile=$1
sets=$2
train_all=$3
#train_all=data/train_all${dataname}
data_dir=`dirname $train_all`

# Train first
for split in ${sets} # train dev test
do
  dirName=${split}${dataname}
  #dirName=callhome_$split
  echo $train_all "is train_all in create_splits"
  echo $data_dir "is data_dir in create_splits"
  cp -r $train_all $data_dir/$dirName

  awk 'BEGIN {FS=" "}; FNR==NR { a[$1]; next } ((substr($2,0,length($2)-2) ".sph") in a)' \
  $splitFile/$split $train_all/segments > $data_dir/$dirName/segments

  n=`awk 'BEGIN {FS = " "}; {print substr($2,0,length($2)-2)}' $data_dir/$dirName/segments | sort | uniq | wc -l`

  echo "$n conversations left in split $dirName"

  utils/fix_data_dir.sh $data_dir/$dirName
  utils/validate_data_dir.sh --no-feats $data_dir/$dirName
done


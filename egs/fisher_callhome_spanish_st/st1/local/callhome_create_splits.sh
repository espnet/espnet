#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

data_dir=data
train_all=data/local/data/callhome_train_all

if [ $# -lt 1 ]; then
    echo "Specify the location of the split files"
    exit 1;
fi

splitFile=$1

# Train first
for split in train devtest evltest; do
    dirName=callhome_$split

    mkdir -p $data_dir/$dirName
    cp $train_all/* $data_dir/$dirName

    awk 'BEGIN {FS=" "}; FNR==NR { a[$1]; next } ((substr($2,0,length($2)-2) ".sph") in a)' \
    $splitFile/$split $train_all/segments > $data_dir/$dirName/segments
    awk 'BEGIN {FS=" "}; FNR==NR { a[$1]; next } ((substr($1,0,length($1)-16) ".sph") in a)' \
    $splitFile/$split $train_all/text > $data_dir/$dirName/text

    n=`awk 'BEGIN {FS = " "}; {print substr($2,0,length($2)-2)}' $data_dir/$dirName/segments | sort | uniq | wc -l`

    echo "$n conversations left in split $dirName"

    # utils/fix_data_dir.sh $data_dir/$dirName
    # utils/validate_data_dir.sh --no-feats $data_dir/$dirName
    # NOTE: do not sort here
done

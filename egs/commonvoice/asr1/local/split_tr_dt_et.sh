#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

perdt=10 # percent for dev set
peret=10 # percent for eval set

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <src-data-dir> <src-dtdata-dir> <src-etdata-dir> <dest-trdata-dir>";
  exit 1;
    exit 1;
fi

sdata=$1
trdata=$4
dtdata=$2
etdata=$3

tmpdata=$trdata/tmp
mkdir -p $tmpdata

# make a unique prompts files and use it to filter out the text&dev set from the validated set
# some transcripts have multiple spaces and need tr -s " " to remove them
cut -f 2- -d" " $etdata/text $dtdata/text | tr -s " " | sort | uniq > $tmpdata/prompts

# it takes very long time when # prompts is large
cat $sdata/text | local/filter_text.py -f $tmpdata/prompts | awk '{print $1}' | sort > $tmpdata/tr.ids
echo "finished filtering with train set from `wc -l $etdata/text | awk '{print $1}'`to `wc -l $tmpdata/tr.ids | awk '{print $1}'`"

reduce_data_dir.sh $sdata $tmpdata/tr.ids $trdata

utils/fix_data_dir.sh $trdata

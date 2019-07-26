#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

perdt=10 # percent for dev set

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <src-data-dir> <dest-trdata-dir> <dest-dtdata-dir> <dest-etdata-dir>";
    exit 1;
fi

sdata=$1
trdata=$2
dtdata=$3

tmpdata=$trdata/tmp
mkdir -p $tmpdata
mkdir -p $dtdata

# make a unique prompts files
# some transcripts have multiple spaces and need tr -s " " to remove them
cut -f 2- -d" " $sdata/text | tr -s " " | sort | uniq > $tmpdata/prompts
num_prompt=`wc -l $tmpdata/prompts | awk '{print $1}'`

num_dt=`echo "$num_prompt * $perdt / 100" | bc`
echo "number of dev set prompts: $num_dt"

# dt
utils/shuffle_list.pl $tmpdata/prompts | head -n $num_dt > $tmpdata/dt_prompts
# tr
nrest=`echo "$num_dt + 1" | bc`
utils/shuffle_list.pl $tmpdata/prompts | \
    tail -n +$nrest > $tmpdata/tr_prompts
echo "number of train set prompts: `wc -l $tmpdata/tr_prompts | awk '{print $1}'`"

# it takes very long time when # prompts is large
cat $sdata/text | local/filter_text.py -f $tmpdata/dt_prompts | awk '{print $1}' | sort > $tmpdata/dt.ids
echo "finished text extraction for dev set #utt = `wc -l $tmpdata/dt.ids | awk '{print $1}'`"
cat $tmpdata/dt.ids | sort > $tmpdata/dtet.ids
cat $sdata/text | awk '{print $1}' | sort > $tmpdata/all.ids
diff $tmpdata/all.ids $tmpdata/dtet.ids | awk '/^</{print $2}' | sort > $tmpdata/tr.ids
echo "finished trans.txt extraction for dev set #utt = `wc -l $tmpdata/tr.ids | awk '{print $1}'`"

reduce_data_dir.sh $sdata $tmpdata/dt.ids $dtdata
reduce_data_dir.sh $sdata $tmpdata/tr.ids $trdata

utils/fix_data_dir.sh $dtdata
utils/fix_data_dir.sh $trdata

#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

maxframes=2000
minframes=10
maxchars=200
minchars=-1
nlsyms=""

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "usage: $0 olddatadir newdatadir"
    exit 1;
fi

sdir=$1
odir=$2
mkdir -p $odir/tmp

echo "remove utterances having more than $maxframes or less than $minframes frames"
feat-to-len scp:$sdir/feats.scp ark,t:- \
    | awk -v maxframes="$maxframes" '{ if ($2 < maxframes) print }' \
    | awk -v minframes="$minframes" '{ if ($2 > minframes) print }' \
    | awk '{print $1}' > $odir/tmp/reclist1

echo "remove utterances having more than $maxchars or less than $minchars characters"
# counting number of chars
if [ -z ${nlsyms} ]; then
text2token.py -s 1 -n 1 $sdir/text \
    | awk -v maxchars="$maxchars" '{ if (NF < maxchars + 1) print }' \
    | awk -v minchars="$minchars" '{ if (NF > minchars + 1) print }' \
    | awk '{print $1}' > $odir/tmp/reclist2
else
text2token.py -l ${nlsyms} -s 1 -n 1 $sdir/text \
    | awk -v maxchars="$maxchars" '{ if (NF < maxchars + 1) print }' \
    | awk -v minchars="$minchars" '{ if (NF > minchars + 1) print }' \
    | awk '{print $1}' > $odir/tmp/reclist2
fi

# extract common lines
comm -12 <(sort $odir/tmp/reclist1) <(sort $odir/tmp/reclist2) > $odir/tmp/reclist

reduce_data_dir.sh $sdir $odir/tmp/reclist $odir
utils/fix_data_dir.sh $odir

oldnum=`wc -l $sdir/feats.scp | awk '{print $1}'`
newnum=`wc -l $odir/feats.scp | awk '{print $1}'`
echo "change from $oldnum to $newnum"

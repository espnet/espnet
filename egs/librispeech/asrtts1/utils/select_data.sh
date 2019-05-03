#!/bin/bash

# This script reads in a data directory with feats.scp and generates a new directory
# with constrained set of features

# begin configuration section
# end configuration section

set -e

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 <inputdir> <outputdir> <no of hours>"
  echo "e.g.:"
  echo " $0 data/train data/train_5hrs 5"
  echo "Options"
  exit 1;
fi

data=$1
out=$2
target_hrs=$3

feat-to-len scp:$data/feats.scp ark,t:- | awk '{sum+=$2} END {print sum/360000}' > tmp.hours
hrs=$(cat tmp.hours)
inthrs=${hrs%.*}
utt=$(cat $data/feats.scp | wc -l)
utt_perhr=$(($utt / $inthrs)) 
intutt_perhr=${utt_perhr%.*}
target_utts=$(($utt_perhr * $target_hrs))
utils/subset_data_dir.sh $data $target_utts $out

#!/bin/bash
. path.sh

data=$1
outdir=$2

feats=(`awk '{print $2}' $data/feats.scp | cut -f1 -d ":"`)
featids=(`awk '{print $1}' $data/feats.scp`)
num=$(echo "${#feats[@]}")
for (( f=0; f<$num; f++ )); do
    echo "reading feat ${feats[$f]}"
    len=$(feat-to-len --print-args=false ark:${feats[$f]} ark,t:- | \
        grep "${featids[$f]}" | awk '{print $2}')
    select-feats ${len}-0 ark:${feats[$f]} ark:$outdir/${featids[$f]}.ark

done

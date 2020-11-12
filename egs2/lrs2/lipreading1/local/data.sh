#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


. ./db.sh
. ./path.sh
. ./cmd.sh


for dataset in train val test; do
    mkdir -p data/${dataset}
    awk  -v lrs2=${LRS2} -F '[/ ]' '{print $1"_"$2, lrs2"/main/"$1"/"$2".mp4"}' ${LRS2}/${dataset}.txt | sort > data/${dataset}/video.scp
    awk '{print $1, "ffmpeg -i " $2 " -ar 16000 -ac 1  -f wav pipe:1 |" }' data/${dataset}/video.scp > data/${dataset}/wav.scp
    awk '{print $2}' data/${dataset}/video.scp | sed -e 's/.mp4/.txt/g' | while read line 
    do 
        grep 'Text:' $line | sed -e 's/Text:  //g'
    done > data/${dataset}/text_tmp
    paste  <(awk '{print $1}' data/${dataset}/wav.scp)  data/${dataset}/text_tmp >  data/${dataset}/text
    rm data/${dataset}/text_tmp
    awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/utt2spk
    awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/spk2utt

done


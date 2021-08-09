#!/bin/bash
DIR="/data1/Code/Sathvik/espnet_mucs/espnet/egs/subtask1/asr1/rawData/Tamil/train/audio/*"
reDir="/data1/Code/Sathvik/espnet_mucs/espnet/egs/subtask1/asr1/rawData/Tamil/train/audio_re/"
for i in $DIR; do
    # echo ${i##*/}
    # echo $i $reDir${i##*/}
    ffmpeg -y  -i "$i" -ar 8000 "$reDir${i##*/}"


done
# for i in *.*; do
#     ffmpeg -i "$i" -your_option "/Volumes/Misc/Converted/${i%.*}.m4v"
# done
#
# ffmpeg -i source.wav -ar 8000 destination.wav

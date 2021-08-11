#!/bin/bash
DIR=$1
reDir=$2

for i in $DIR; do
    ffmpeg -y  -i "$i" -ar 8000 "$reDir${i##*/}"
done

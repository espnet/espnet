#!/bin/bash

#local/prepare_data.sh data/hi-en/test/transcripts/wav.scp data/hi-en/test/audio/ out.scp

IFS=$'\n'
set -f
for i in $(cat < "$1"); do
   stem=$( echo $i | cut -d' ' -f1)
   path=$2$( echo $i | cut -d' ' -f2)
   echo $stem $path  >> $3
done
mv $3 $1

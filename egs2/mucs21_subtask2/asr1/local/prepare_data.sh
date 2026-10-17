#!/bin/bash

IFS=$'\n'
set -f
for i in $(cat < "$1"); do
   stem=$( echo $i | cut -d' ' -f1)
   path=$2$( echo $i | cut -d' ' -f2)
   echo $stem $path  >> $3
done
mv $3 $1

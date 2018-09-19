#! /bin/bash


if [ $# -ne 2 ]; then
  echo "Usage: ./view_lm_data.sh <data> <num_utts>"
  exit 1;
fi

data=$1
num_utts=$2

awk -v var=${num_utts} '{i=0; j=1; while(i < var){ if($j == "<eos>"){i+=1; printf("%s\n", $j); j+=1} else {printf("%s ", $j); j+=1}}}' $data



#!/bin/bash

. ./path.sh
. ./cmd

if [ $# -ne 1 ]; then
  echo "Usage: ./local/prepare_librilight.sh <data-path>"
  exit 1;
fi

data=$1
find -L $data -name "*.flac"

for part in 1h/{0..5}/{clean,other} 9h/{clean,other}; do
    dataname=$(echo ${part} | sed 's/\//_/g')
    
    data_part=$(./utils/make_absolute.sh ${dataname})
    data_new_path=data/train_${dataname}
    mkdir -p ${data_new_path}
    files=( `find -L ${data_part}/ -name "*.flac"` )
    
    for f in ${files[@]}; do
	filename=`basename $f`
	filename=${filename%%.flac}
	echo "${fname} flac -c -d -s ${f} |" 
    done | sort > ${data_new_path}/wav.scp
    
    paste -d' ' <(awk '{print $1}' ${data_new_path}/wav.scp) \
          <(awk '{print $1}' ${data_new_path}/wav.scp | cut -d'-' -f1) \
          > ${data_new_path}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl ${data_new_path}/utt2spk > ${data_new_path}/spk2utt
    cat `find -L ${data_part} -name "*.trans.txt"` | sort > ${data_new_path}/text
done

./utils/combine_data.sh \
  data/train_10h data/train_1h_{0..5}_{clean,other} data/train_9h_{clean,other}

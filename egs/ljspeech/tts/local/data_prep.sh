#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${filename}" >> ${scp}
    echo "${id} LJ" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

echo "finished making wav.scp, utt2spk, spk2utt."

# make text
cat ${db}/metadata.csv | while read -r line;do
    id=$(echo "${line}" | awk -F"|" '{print $1}')
    content=$(echo "${line}" | awk -F"|" '{print $2}')
    clean_content=$(PYTHONPATH=local/text local/clean_text.py "${content}")
    echo "${id} ${clean_content}" >> ${text}
done

echo "finished making text."

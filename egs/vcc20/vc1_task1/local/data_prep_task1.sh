#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
spk=$3
trans_type=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db> <data_dir> <spk> <trans_type>"
    exit 1
fi

# check speaker
available_spks=(
    "TEF1" "TEF2" "TEM1" "TEM2"
)

if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified speaker ${spk} is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

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
find ${db}/${spk} -follow -name "*.wav" | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text (only for the utts in utt2spk)
while IFS= read -r line; do
    id=$(echo ${line} | cut -d " " -f1)
    num=$(echo ${id} | tail -c 6)
    grep ${num} ${db}/prompts/Eng_transcriptions.txt \
        | sed -e "s/${num}/${id}/g" >> ${text}
done < ${utt2spk}
echo "finished making text."

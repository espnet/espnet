#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. utils/parse_options.sh || exit 1

db=$1
data_dir=$2
lang=$3
spk=csmsc

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <lang>"
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
find ${db}/Wave -name "*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} .wav)"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
find ${db}/PhoneLabeling -name "*.interval" -follow | sort | while read -r filename;do
    id="$(basename ${filename} .interval)"
    content=$(tail -n +13 ${filename} | grep "\"" | grep -v "sil" | sed -e "s/\"//g" | tr "\n" " " | sed -e "s/ $//g")
    start_sec=$(tail -n +14 ${filename} | head -n 1)
    end_sec=$(head -n -2 ${filename} | tail -n 1)
    echo "${spk}_${id} <${lang}> ${content}" >> ${text}
done
echo "Successfully finished making text. One of the segment info is wrong so let's not to use it."

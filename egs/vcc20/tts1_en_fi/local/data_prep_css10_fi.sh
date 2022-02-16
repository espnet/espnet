#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. utils/parse_options.sh || exit 1

db=$1
data_dir=$2
lang=$3
spk=$4
trans_type=char

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db> <data_dir> <lang> <spk>"
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

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" -follow | grep ${spk} | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate 16000 |" >> "${scp}"
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

lang_tag=${lang}
local/clean_text_css10.py \
    --lang_tag ${lang_tag} \
    --spk_tag ${spk} \
    ${db}/transcript.txt \
    ${data_dir}/text \
    ${trans_type}
echo "Successfully finished making text."

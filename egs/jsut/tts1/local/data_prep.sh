#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Takenori Yoshimura), Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
input_type=$3

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
rawtext=${data_dir}/rawtext
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${rawtext} ] && rm ${rawtext}

# make scp, utt2spk, and spk2utt
find ${db} -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${filename}" >> ${scp}
    echo "${id} JS" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text
find ${db} -follow -name "transcript_utf8.txt" | sort | while read -r filename; do
    cat ${filename} >> ${rawtext}
done
local/clean_text.py \
    ${rawtext} ${text} ${input_type}
rm ${rawtext}
echo "finished making text."

local/prep_segments.py $scp > ${data_dir}/segments
echo "finished making segments."

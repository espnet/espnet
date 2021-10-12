#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db> <data_dir>"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text
segments=${data_dir}/segments

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}
[ -e ${segments} ] && rm ${segments}

# make scp, utt2spk, and spk2utt
find ${db}/Wave -name "*.wav" -follow | sort | while read -r filename;do
    id="$(basename ${filename} .wav)"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} csmsc" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text and segments
find ${db}/PhoneLabeling -name "*.interval" -follow | sort | while read -r filename;do
    id="$(basename ${filename} .interval)"
    content=$(nkf -Lu -w ${filename} | tail -n +13 | grep "\"" | grep -v "sil" | sed -e "s/\"//g" | tr "\n" " " | sed -e "s/ $//g")
    start_sec=$(nkf -Lu -w ${filename} | tail -n +14 | head -n 1)
    end_sec=$(nkf -Lu -w ${filename} | head -n -2 | tail -n 1)
    echo "${id} ${content}" >> ${text}
    echo "${id} ${id} ${start_sec} ${end_sec}" >> ${segments}
done
echo "Successfully finished making text, segments."

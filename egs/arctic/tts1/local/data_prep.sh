#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <spk> <data_dir>"
    exit 1
fi

# check speaker
available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
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
segments=${data_dir}/segments

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${segments} ] && rm ${segments}

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
raw_text=${db}/etc/arctic.data
ids=$(sed < ${raw_text} -e "s/^( /${spk}_/g" -e "s/ )$//g" | cut -d " " -f 1)
sentences=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" | tr '[:lower:]' '[:upper:]' | cut -d " " -f 2-)
paste -d " " <(echo "${ids}") <(echo "${sentences}") > ${text}.tmp
local/clean_text.py ${text}.tmp > ${text}
rm ${text}.tmp
echo "Successfully finished making text."

# make segments
find ${db}/lab -name "*.lab" -follow | sort | while read -r filename; do
    # get start time
    while read line; do
        phn=$(echo ${line} | cut -d " " -f 3)
        if [ ${phn} != "pau" ]; then
            break
        fi
        start=$(echo ${line} | cut -d " " -f 1)
    done < <(tail -n +2 $filename)
    # get end time
    while read line; do
        end=$(echo ${line} | cut -d " " -f 1)
        phn=$(echo ${line} | cut -d " " -f 3)
        if [ ${phn} != "pau" ]; then
            break
        fi
    done < <(tail -n +2 $filename | tac)
    echo "${spk}_$(basename ${filename} .lab) ${spk}_$(basename ${filename} .lab) ${start} ${end}" >> ${segments}
done
echo "Successfully finished making segments."

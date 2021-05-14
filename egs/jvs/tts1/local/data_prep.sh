#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Takenori Yoshimura), Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2
org_data_dir=$3
input_type=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <corpus_dir> <target_spk> <data_dir> <input_type: char or phn>"
    exit 1
fi

# check spk existence
[ ! -e ${db}/${spk} ] && echo "${spk} does not exist." >&2 && exit 1;

# NOTE: we use only parallel100 and nonpara30 for TTS
for name in parallel100 nonpara30; do
    data_dir=${org_data_dir}_${name}

    # check directory existence
    [ ! -e ${data_dir} ] && mkdir -p ${data_dir}

    # set filenames
    scp=${data_dir}/wav.scp
    utt2spk=${data_dir}/utt2spk
    spk2utt=${data_dir}/spk2utt
    rawtext=${data_dir}/rawtext
    text=${data_dir}/text
    segments=${data_dir}/segments

    # check file existence
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${rawtext} ] && rm ${rawtext}
    [ -e ${segments} ] && rm ${segments}

    # make scp, utt2spk, and spk2utt
    find ${db}/${spk}/${name} -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${spk}_${id} ${filename}" >> ${scp}
        echo "${spk}_${id} ${spk}" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
    echo "finished making wav.scp, utt2spk, spk2utt."

    # make text
    find ${db}/${spk}/${name} -follow -name "transcripts_utf8.txt" | sort | while read -r filename; do
        cat ${filename} >> ${rawtext}
    done
    local/clean_text.py \
        ${rawtext} ${text} ${input_type}
    sed -e "s/^/${spk}_/g" -i ${text}
    rm ${rawtext}
    echo "finished making text."

    # make segments
    find ${db}/${spk}/${name}/lab/mon -follow -name "*.lab" | sort | while read -r filename; do
        start=$(head -n 1 ${filename} | cut -d " " -f 2)
        end=$(tail -n 1 ${filename} | cut -d " " -f 1)
        id="${spk}_$(basename ${filename} .lab)"
        echo "${id} ${id} ${start} ${end}" >> ${segments}
    done
    echo "finished making segments."

    # check
    utils/fix_data_dir.sh ${data_dir}
done
echo "Successfully finished data preparation."

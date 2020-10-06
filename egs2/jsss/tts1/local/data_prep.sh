#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Prepare kaldi-style data directory for JSSS corpus

db=$1
data_dir=$2
fs=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <fs>"
    echo "e.g.: $0 downloads/jsss_ver1 data/all 24000"
    exit 1
fi

set -euo pipefail

data_dir_root=$(dirname "${data_dir}")

# process data without segmetns
dsets_without_segments="
short-form/basic5000
short-form/onomatopee300
short-form/voiceactress100
simplification
"
for dset in ${dsets_without_segments}; do
    # check directory existence
    _data_dir=${data_dir_root}/$(basename "${dset}")
    [ ! -e "${_data_dir}" ] && mkdir -p "${_data_dir}"

    # set filenames
    scp=${_data_dir}/wav.scp
    utt2spk=${_data_dir}/utt2spk
    spk2utt=${_data_dir}/spk2utt
    text=${_data_dir}/text

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"

    # make scp, utt2spk, and spk2utt
    find "${db}/${dset}" -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        if [ "${fs}" -eq 24000 ]; then
            # default sampling rate
            echo "${id} ${filename}" >> "${scp}"
        else
            echo "${id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
        fi
        echo "${id} JSSS" >> "${utt2spk}"
    done
    utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

    # make text
    find "${db}/${dset}" -name "transcript_utf8.txt" | sort | while read -r filename; do
        tr ':' ' ' < "${filename}" | sort >> "${text}"
    done

    # fix
    utils/fix_data_dir.sh "${_data_dir}"
    echo "Successfully prepared ${dset}."
done

# process data with segmetns
dsets_with_segments="
long-form/katsura-masakazu
long-form/udon
long-form/washington-dc
"
for dset in ${dsets_with_segments}; do
    # check directory existence
    _data_dir=${data_dir_root}/$(basename "${dset}")
    [ ! -e "${_data_dir}" ] && mkdir -p "${_data_dir}"

    # set filenames
    scp=${_data_dir}/wav.scp
    utt2spk=${_data_dir}/utt2spk
    spk2utt=${_data_dir}/spk2utt
    text=${_data_dir}/text
    segments=${_data_dir}/segments

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"
    [ -e "${segments}" ] && rm "${segments}"

    # make wav.scp, utt2spk, and spk2utt
    find "${db}/${dset}" -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        if [ "${fs}" -eq 24000 ]; then
            # default sampling rate
            echo "${id} ${filename}" >> "${scp}"
        else
            echo "${id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
        fi
    done
    # make utt2spk, spk2utt, and text
    find "${db}/${dset}/transcript_utf8" -name "*.txt" | sort | while read -r filename; do
        wav_id=$(basename "${filename}" .txt)
        while read -r line; do
            start_sec=$(echo "${line}" | cut -f 1)
            end_sec=$(echo "${line}" | cut -f 2)
            sentence=$(echo "${line}" | cut -f 3)
            utt_id=${wav_id}
            utt_id+="_$(printf %010d "$(echo "${start_sec}" | tr -d "." | sed -e "s/^[0]*//g")")"
            utt_id+="_$(printf %010d "$(echo "${end_sec}" | tr -d "." | sed -e "s/^[0]*//g")")"
            echo "${utt_id}" "${sentence}" >> "${text}"
            echo "${utt_id} JSSS" >> "${utt2spk}"
            echo "${utt_id} ${wav_id} ${start_sec} ${end_sec}" >> ${segments}
        done < "${filename}"
    done
    utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

    # fix
    utils/fix_data_dir.sh "${_data_dir}"
    echo "Successfully prepared ${dset}."
done

# combine all data
combined_data_dirs=""
for dset in ${dsets_without_segments} ${dsets_with_segments}; do
    combined_data_dirs+="${data_dir_root}/$(basename ${dset}) "
done
utils/combine_data.sh "${data_dir}" ${combined_data_dirs}
rm -rf ${combined_data_dirs}

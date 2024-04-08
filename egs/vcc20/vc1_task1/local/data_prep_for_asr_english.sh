#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

wavdir=$1
data_dir=$2
spk=$3
list=$4
transcription=$5

# check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <wavdir> <data_dir> <spk> <list> [transcription]"
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
while read f; do
    filename=${wavdir}/${f}.wav
    if [[ -e ${filename} ]]; then
        echo "${spk}_${f} ffmpeg -loglevel warning -i ${filename} -ac 1 -ar 16000 -acodec pcm_s16le -f wav -y - |" >> ${scp}
        echo "${spk}_${f} ${spk}" >> ${utt2spk}
    fi
done < ${list}
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

if [ ! -z ${transcription} ]; then
    # make text
    local/clean_text_english.py ${transcription} ${utt2spk} char > ${text}
    echo "finished making text."

    # remove reduntant lines of text
    utils/fix_data_dir.sh ${data_dir}
else
    # make dump text
    while read f; do
        filename=${wavdir}/${f}.wav
        if [[ -e ${filename} ]]; then
            echo "${spk}_${f} ." >> ${text}
        fi
    done < ${list}
fi

#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
metadata=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <metadata>"
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
sox=`which sox` || { echo "Could not find sox in PATH"; exit 1; }
silence="local/ob_eval/silence_16k_100ms.wav"
find ${db} -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} $sox -t wav ${filename} -c 1 -b 16 -t wav - rate 16000 | $sox ${silence} -t wav - -t wav - |" >> ${scp}
    echo "${id} LJ" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text
local/clean_text.py ${metadata} char > ${text}
echo "finished making text."

# remove reduntant lines of text
utils/fix_data_dir.sh ${data_dir}

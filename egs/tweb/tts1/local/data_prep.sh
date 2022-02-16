#!/usr/bin/env bash
set -e

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db_root> <data_dir>"
    exit 1
fi

# check db existence
if [ ! -e ${db}/README.md ]; then
    echo "It seems that database path is not correctly set." >&2
    echo "If you have not yet downloaded database, please follow the instruction in egs/tweb/README.md" >&2
    exit 1;
fi

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

# some samples are stereo so we use sox to convert monaural
sox=`which sox` || { echo "Could not find sox in PATH"; exit 1; }

# make scp, utt2spk, and spk2utt
find ${db} -maxdepth 2 -name "*.wav" -follow | sort | while read -r filename;do
    id="$(echo $(basename ${filename} .wav) | tr '()' '__')"
    echo "${id} $sox \"${filename}\" -c 1 -b 16 -t wav - |" >> ${scp}
    echo "${id} tweb" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
cat ${db}/transcript.txt | cut -d "/" -f 2- | sed -e "s/\t/ /g" > ${rawtext}
local/clean_text.py ${rawtext} > ${text}
rm ${rawtext}
echo "Successfully finished making text."

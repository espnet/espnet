#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
trgspk=$3

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
find ${db} -name "*.wav" | sort | while read -r filename;do

    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g" | sed -e "s/-feats//g" | sed -e "s/_gen//g" | sed -e "s/${trgspk}_//g" )
    id=${trgspk}_$id

    echo "${id} ffmpeg -loglevel warning -i ${filename} -ac 1 -ar 16000 -acodec pcm_s16le -f wav -y - |" >> ${scp}
    echo "${id} $trgspk" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

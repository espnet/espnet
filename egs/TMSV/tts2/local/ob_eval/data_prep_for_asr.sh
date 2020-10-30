#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

wav_dir=$1
data_dir=$2
set_name=$3

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
#[ -e ${text} ] && rm ${text}
[ -e ${utt2spk} ] && rm ${utt2spk}

# make scp, utt2spk, and spk2utt
find ${wav_dir} -name "*.wav" | sort | while read -r filename;do
    
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g" | sed -e "s/-feats//g" | sed -e "s/_gen//g"  )
    id=pwg_${id}
    
    echo "${id} ${filename}" >> ${scp}
done


#sed "s/^/pwg_&/g" data/${set_name}/text > ${data_dir}/text
# due to we use different dictionary (phone-based) with the aishell (character based, btw its use simple cymbol), now we 
# generate the text file by the most stupid way: our hand.
sed "s/^/pwg_&/g" data/${set_name}/utt2spk > ${utt2spk}

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."
echo -e "\033[33mDon't forget to generate the text file (${text}) yourself  \033[0m"

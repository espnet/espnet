#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_name=$2
kaldi_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_name> <kaldi_dir>"
    exit 1
fi

# check directory existence
[ ! -e ${kaldi_dir} ] && mkdir -p ${kaldi_dir}

# set filenames
scp=${kaldi_dir}/wav.scp
utt2spk=${kaldi_dir}/utt2spk
id2spk=${kaldi_dir}/id2spk
spk2utt=${kaldi_dir}/spk2utt
text=${kaldi_dir}/text
text_slu=${kaldi_dir}/text.slu
segments=${kaldi_dir}/segments
utt2dur=${kaldi_dir}/utt2dur

db=$(readlink -f $db)
data_name=${data_name}_data.csv
kaldi_dir=$(readlink -f $kaldi_dir)
# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${spk2utt} ] && rm ${spk2utt}
[ -e ${text} ] && rm ${text}
[ -e ${text_slu} ] && rm ${text_slu}
[ -e ${segments} ] && rm ${segments}
[ -e ${id2spk} ] && rm ${id2spk}

#    echo "${speakerID} ${utt}" >> ${spk2utt}

while IFS=, read -r id path spkID remaining; do
    #speakerID=$(echo spk$spkID | awk '{s=$0; while(length(s)<21) s=s "z"; print s}')
    speakerID=$(echo spk_$spkID)
    #id=$(echo utt${c} | awk '{s=$0; while(length(s)<8) s=s "z"; print s}')
    utt=$(echo ${speakerID}-${id})
    echo "processing $speakerID $utt"
    txt=$(echo ${remaining} | awk -F, 'NF-=3')
    slu=$(echo ${remaining} | awk -F, '{print $(NF-2),$(NF-1),$(NF)}' | tr " " "_")
    echo "${utt} $db/${path}" >> ${scp}
    echo "${utt} ${speakerID}" >> ${utt2spk}
    echo "${utt} ${txt}" >> ${text}
    echo "${utt} ${slu}" >> ${text_slu}
    echo "${utt} ${utt} 0" >> ${segments}.tmp
done < <(tail -n +2 $db/data/$data_name)

wav-to-duration scp:${scp} ark,t:${utt2dur}
paste -d' ' ${segments}.tmp <(cut -d' ' -f2 ${utt2dur}) > ${segments}
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
rm ${segments}.tmp
#echo "utils/fix_data_dir.sh --utt_extra_files "text.slu" ${kaldi_dir}"
#utils/fix_data_dir.sh --utt_extra_files "text.slu" ${kaldi_dir}

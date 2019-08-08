#!/bin/bash 

# Copyright 2018 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
fs=44100

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

# make downsampled dir
cp -a ${db}/enUK ${db}/enUK.rename ; echo "db have been copied." 

# replace ' ' of the filename to '_'
find ${db}/enUK.rename -name "*.m4a" | rename 's/ /_/g'
find ${db}/enUK.rename -name "*.mp3" | rename 's/ /_/g'
find ${db}/enUK.rename -name "*.wma" | rename 's/ /_/g'
find ${db}/enUK.rename -name "*.lab" | rename 's/ /_/g'

# edit mistakes
mkdir -p local/tmp
edit_file=${db}/enUK.rename/fls/WindInTheWillows_picturebook/lab/Wind_in_Willows_final01.lab
cat ${edit_file} | awk '{if(NR==27){printf("%s poop\n",$0);}else{print $0;}}' > local/tmp/tmp.txt
cat local/tmp/tmp.txt > ${edit_file}

# make new lab file from txt
local/make_new_lab.sh ${db}/enUK.rename

# make scp
echo -n > local/tmp/tmp.scp
for ftype in "m4a" "mp3" "wma";do
    find ${db}/enUK.rename -name "*.${ftype}" | sort | while read -r in_file;do
	lab_file=$(echo ${in_file} | sed -e "s/audio/new_lab/g" -e "s/${ftype}/lab/g")
	if [ -e ${lab_file} ]; then
	    id=$(basename ${in_file} | sed -e "s/\.[^\.]*$//g")
	    dir_id=$(echo ${in_file} | awk -F'/' '{print $(NF-2)}')
	    echo "${dir_id}_${id} ffmpeg -loglevel warning -i ${in_file} -ac 1 -ar ${fs} -acodec pcm_s16le -f wav -y - |" >> local/tmp/tmp.scp
	fi
    done
done
cat local/tmp/tmp.scp | sort > ${scp}
echo "finished making wav.scp."

# make segments, text
echo -n > ${segments}
echo -n > ${text}
cat ${scp} | awk '{print $6}' | sed -e "s/audio/new_lab/g" -e "s/mp3/lab/g" -e "s/m4a/lab/g" -e "s/wma/lab/g" | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    dir_id=$(echo ${filename} | awk -F'/' '{print $(NF-2)}')
    cat ${filename} | awk -v "utt_id=${dir_id}_${id}" \
    '{if($3!="#") printf("%s_%06d-%06d %s %f %f\n",utt_id,$1*100,$2*100,utt_id,$1,$2)}' >> ${segments}
    cat ${filename} | awk -v "utt_id=${dir_id}_${id}" \
    '{if($3!="#"){ printf("%s_%06d-%06d ",utt_id,$1*100,$2*100); for(i=3;i<=NF;i++){printf("%s ",$i);} printf("\n");}}' >> ${text}
done
echo "finished making segments, text."

# make utt2spk, spk2utt
cat ${segments} | awk '{printf("%s blizzard\n",$1);}' > ${utt2spk}
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making utt2spk, spk2utt."

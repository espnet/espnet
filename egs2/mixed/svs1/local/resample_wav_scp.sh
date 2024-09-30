#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: $0 <frequency> <src_wav_scp> <dst_wav_dump_dir> <dst_wav_scp>"
  echo " e.g.: $0 16000 data/dev/wav.scp wav_dump_16k data_16k/dev/wav.scp"
  exit 1
fi

fs=$1
src_wav_scp=$2
dst_wav_dump_dir=$3
dst_wav_scp=$4

mkdir -p ${dst_wav_dump_dir}
mkdir -p $(dirname ${dst_wav_scp})

# iterate over all lines in the wav.scp file
while IFS=" " read -r utt src_file; do
    if [[ $(soxi -r ${src_file}) -eq ${fs} ]]; then
        echo "${utt} ${src_file}" >> ${dst_wav_scp}
    else
        dst_file=${dst_wav_dump_dir}/${utt}.wav
        dst_file=$(realpath ${dst_file})
        sox ${src_file} -r ${fs} ${dst_file}
        echo "${utt} ${dst_file}" >> ${dst_wav_scp}
    fi
done < ${src_wav_scp}

#!/bin/bash -e

# Copyright 2019 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 4 ]; then
    echo "Num of arguments should be 4"
    exit 1 
fi

out_wavdir=$1
gt_wavdir=$2
out_spk_wavdir=$3
gt_spk_wavdir=$4

spk=$(basename $out_spk_wavdir)

for out_wav_file in $(find -L ${out_wavdir} -iname "${spk}_*" | sort ); do

    wav_basename=$(basename $out_wav_file .wav)
    echo "${wav_basename}"

    cp ${out_wav_file} ${out_spk_wavdir} || exit 1
    #mv ${out_spk_wavdir}/${wav_basename}_gen.wav ${out_spk_wavdir}/${wav_basename}.wav
    cp ${gt_wavdir}/${wav_basename}.wav ${gt_spk_wavdir}

done

echo "Succeessfully create ${spk}'s directories for mcd evaluation"

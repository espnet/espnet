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

for i in $(seq 316 1 320); do
    out_wav_file=${out_wavdir}/${spk}_${i}_gen.wav
    cp ${out_wav_file} ${out_spk_wavdir} || exit 1
    mv ${out_spk_wavdir}/${spk}_${i}_gen.wav ${out_spk_wavdir}/${spk}_${i}.wav
    gt_wav_file=${gt_wavdir}/${spk}_${i}.wav
    cp ${gt_wav_file} ${gt_spk_wavdir} || exit 1
done

echo "Succeessfully create ${spk}'s directories for mcd evaluation"

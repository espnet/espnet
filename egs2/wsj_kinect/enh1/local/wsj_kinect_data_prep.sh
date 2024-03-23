#!/usr/bin/env bash

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

set -e
set -u
set -o pipefail

min_or_max="min"
sample_rate="16k"

. utils/parse_options.sh
. ./path.sh

if [ $# -le 1 ]; then
  echo "Arguments should be 2speakers_reverb_kinect path and the Reverberated_WSJ_2MIX/list/ path"
  exit 1;
fi


wavdir="$1/2speakers_reverb_kinect_chime_noise_corrected/wav${sample_rate}/${min_or_max}"
listdir=$2
destdir=./data

tr="tr"
cv="cv"
tt="tt"

# Ensure that the wav dir exists
for f in "$wavdir/$tr" "$wavdir/$cv" "$wavdir/$tt"; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# Ensure that the file list exists
for f in ${listdir}/wsj0-2mix_tr.flist ${listdir}/wsj0-2mix_cv.flist ${listdir}/wsj0-2mix_tt.flist ; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done


for x in tr cv tt; do
  target_folder=$x
  mkdir -p ${destdir}/$target_folder
  ls -1 "${wavdir}/$x/mix" | \
    awk  '{split($1, lst, ".wav"); print(lst[1])}' | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${destdir}/${target_folder}/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${destdir}/${target_folder}/wav.scp | sort > ${destdir}/${target_folder}/utt2spk
  utils/utt2spk_to_spk2utt.pl ${destdir}/${target_folder}/utt2spk > ${destdir}/${target_folder}/spk2utt
done

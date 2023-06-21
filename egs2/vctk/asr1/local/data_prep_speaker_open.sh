#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Arguments should be NOISY_SPEECH wav path, see local/data.sh for example."
  exit 1;
fi

NOISY_SPEECH=$1
# check if the wav dirs exist.

for ddir in clean_trainset_28spk_wav clean_testset_wav trainset_28spk_txt testset_txt; do
  f=${NOISY_SPEECH}/${ddir}
  if [ ! -d $f ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done

data=./data
rm -r ${data}/tr_26spk 2>/dev/null || true
rm -r ${data}/{cv, tt}_2spk 2>/dev/null || true

tmpdir=data/temp
rm -r  $tmpdir 2>/dev/null || true
mkdir -p $tmpdir 

train_dir=${NOISY_SPEECH}/clean_trainset_28spk_wav
test_dir=${NOISY_SPEECH}/clean_testset_wav

echo "Building training and testing data"

find $train_dir -name '*.wav' -not -name 'p226_*.wav' -not -name 'p287_*.wav' | sort -u > $tmpdir/tr_no_dev.flist
find $train_dir -name 'p226_*.wav' -o -name 'p287_*.wav' | sort -u > $tmpdir/dev.flist
find $test_dir -name '*.wav' | sort -u > $tmpdir/eval1.flist


for x in tr_no_dev dev eval1; do

  if [ "${x}" == "tr_no_dev" -o "${x}" == "dev" ]; then
      text_dir=${NOISY_SPEECH}/trainset_28spk_txt
  else
      text_dir=${NOISY_SPEECH}/testset_txt
  fi      

  sed -e 's:.*p\([0-9]*\)_\([0-9]*\).wav$:p\1_\2:i' $tmpdir/${x}.flist \
  > $tmpdir/${x}.uttids

  paste $tmpdir/${x}.uttids $tmpdir/${x}.flist \
  | sort -k1,1 >  $tmpdir/${x}.scp
  mkdir -p ${data}/${x}
  cp $tmpdir/${x}.scp ${data}/${x}/wav.scp
  
  awk '{split($1, lst, "_"); spk=lst[1]; print($1, spk)}' ${data}/${x}/wav.scp | \
    sort -u> ${data}/${x}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${x}/utt2spk > ${data}/${x}/spk2utt

  cat $tmpdir/${x}.uttids | \
      while read uttid;
      do
	  if [ ! -f ${text_dir}/${uttid}.txt ]; then
	      echo "missing text file for ${uttid}" 1>&2
	      exit 1;
	  fi
	  echo "${uttid}" $(<${text_dir}/${uttid}.txt)
      done | \
	  sort -u > ${data}/${x}/text

  sed -e "s#noisy_#clean_#g" ${data}/${x}/wav.scp \
    > ${data}/${x}/spk1.scp
done




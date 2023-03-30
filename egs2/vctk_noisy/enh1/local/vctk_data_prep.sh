#!/usr/bin/env bash

. ./path.sh



if [ $# -ne 1 ]; then
  echo "Arguments should be NOISY_SPEECH wav path, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

NOISY_SPEECH=$1
# check if the wav dirs exist.

for ddir in clean_trainset_28spk_wav noisy_trainset_28spk_wav clean_testset_wav noisy_testset_wav; do
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

train_dir=${NOISY_SPEECH}/noisy_trainset_28spk_wav
test_dir=${NOISY_SPEECH}/noisy_testset_wav

echo "Building training and testing data"

find $train_dir -name '*.wav' -not -name 'p226_*.wav' -not -name 'p287_*.wav' | sort -u > $tmpdir/tr.flist
find $train_dir -name 'p226_*.wav' -o -name 'p287_*.wav' | sort -u > $tmpdir/cv.flist
find $test_dir -name '*.wav' | sort -u > $tmpdir/tt.flist


for x in tr cv tt; do
  if [ "$x" = "tr" ]; then
    ddir=${x}_26spk
  elif [ "$x" = "cv" -o "$x" = "tt" ]; then
    ddir=${x}_2spk
  fi

  sed -e 's:.*p\([0-9]*\)_\([0-9]*\).wav$:p\1_\2:i' $tmpdir/${x}.flist \
  > $tmpdir/${x}.uttids

  paste $tmpdir/${x}.uttids $tmpdir/${x}.flist \
  | sort -k1,1 >  $tmpdir/${x}.scp
  mkdir -p ${data}/${ddir}
  cp $tmpdir/${x}.scp ${data}/${ddir}/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

  awk '{print($1, "dummy")}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/text

  sed -e "s#noisy_#clean_#g" ${data}/${ddir}/wav.scp \
    > ${data}/${ddir}/spk1.scp
done

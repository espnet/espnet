#!/usr/bin/env bash

. utils/parse_options.sh
. ./path.sh



if [ $# -ne 2 ]; then
  echo "Arguments should be DNS script path, DNS wav path and DNS data, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dns_wav=$1
dns_test_wav=$2

# check if the wav dirs exist.

for ddir in clean noise noisy; do
  if [ ! -d ${dns_wav}/${ddir} ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done

#Synthetic test data
for rev in with_reverb no_reverb; do
  for ddir in clean noisy; do
    f=${dns_test_wav}/synthetic/${rev}/${ddir}
    if [ ! -d $f ]; then
      echo "Error: $f is not a directory."
      exit 1;
    fi
  done
done

#Real_recordings test data
f=${dns_test_wav}/real_recordings
if [ ! -d $f ]; then
  echo "Error: $f is not a directory."
  exit 1;
fi

data=./data
rm -r ${data}/{tr, cv}_synthetic 2>/dev/null || true
rm -r ${data}/tt_synthetic_{with,no}_reverb 2>/dev/null || true

tmpdir=data/temp
rm -r  $tmpdir 2>/dev/null || true
mkdir -p $tmpdir 

mixwav_dir=${dns_wav}/noisy

find $mixwav_dir -iname '*.wav' | sort -u > $tmpdir/train_valid.flist

sed -e 's:.*reader_\([0-9]*\)_.*_\([0-9]*\).wav$:reader_\1_fileid_\2:i' $tmpdir/train_valid.flist \
> $tmpdir/train_valid.uttids

paste $tmpdir/train_valid.uttids $tmpdir/train_valid.flist \
| sort -k1,1 >  $tmpdir/train_valid.scp

num=$(wc -l $tmpdir/train_valid.scp | awk '{print $1}')

echo "Split 10% of the Training data to the Validation data"
cat $tmpdir/train_valid.scp | head -n $(($num*9/10)) > $tmpdir/tr.scp
cat $tmpdir/train_valid.scp | tail -n $(($num - $num*9/10)) > $tmpdir/cv.scp


for x in tr cv; do
  ddir=${x}_synthetic
  mkdir -p ${data}/${ddir}
  cp $tmpdir/${x}.scp ${data}/${ddir}/wav.scp
  
  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

  noise_wav_dir=${dns_wav}/noise/
  sed -e "s#${mixwav_dir}.*_\(.*\).wav#${noise_wav_dir}noise_fileid_\1.wav#g" ${data}/${ddir}/wav.scp \
    > ${data}/${ddir}/noise1.scp

  spk1_wav_dir=${dns_wav}/clean/
  sed -e "s#${mixwav_dir}.*_\(.*\).wav#${spk1_wav_dir}clean_fileid_\1.wav#g" ${data}/${ddir}/wav.scp \
    > ${data}/${ddir}/spk1.scp
done


echo "Building testing data"

for x in tt; do
  for rev in with_reverb no_reverb; do
    ddir=${x}_synthetic_${rev}
    mkdir -p ${data}/${ddir}
    root_dir=${dns_test_wav}/synthetic/${rev}

    mixwav_dir=${root_dir}/noisy/
    find $mixwav_dir -iname '*.wav' > $tmpdir/${x}_${rev}.flist

    sed -e 's:.*_\([0-9]*\).wav$:fileid_\1:i' $tmpdir/${x}_${rev}.flist \
    > $tmpdir/${x}_${rev}.uttids

    paste $tmpdir/${x}_${rev}.uttids $tmpdir/${x}_${rev}.flist \
    | sort -k1,1 >  ${data}/${ddir}/wav.scp 


    awk '{print($1, $1)}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/utt2spk

    utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt
    touch ${data}/${ddir}/text

    spk1_wav_dir=${root_dir}/clean/
    sed -e "s#${mixwav_dir}.*_\(.*\).wav#${spk1_wav_dir}clean_fileid_\1.wav#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/spk1.scp
  done
done


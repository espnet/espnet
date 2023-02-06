#!/usr/bin/env bash

. ./path.sh


if [ $# -ne 1 ]; then
  echo "Arguments should be DNS script path, DNS wav path and DNS data, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dns_wav=$1

# check if the wav dirs exist.
for ddir in clean noise noisy; do
  f=${dns_wav}/${ddir}
  if [ ! -d $f ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done

data=./data
rm -r ${data}/{tr, cv}_synthetic 2>/dev/null || true
rm -r ${data}/tt_synthetic 2>/dev/null || true

tmpdir=data/temp
rm -r  $tmpdir 2>/dev/null || true
mkdir -p $tmpdir 

mixwav_dir=${dns_wav}/noisy

find $mixwav_dir -iname '*.wav' | sort -u > $tmpdir/train_valid.flist

sed -e 's:.*_\([0-9]*\).wav$:fileid_\1:i' $tmpdir/train_valid.flist \
> $tmpdir/train_valid.uttids

paste $tmpdir/train_valid.uttids $tmpdir/train_valid.flist \
| sort -k1,1 >  $tmpdir/train_valid.scp

num=$(wc -l $tmpdir/train_valid.scp | awk '{print $1}')
train_num=$(($num*8/10))
train_val_num=$(($num*9/10))

echo "Split 10%, 10% of the Training data to the Validation set and Test set"
awk "NR<=$train_num" $tmpdir/train_valid.scp > $tmpdir/tr.scp
awk "(NR>$train_num) && (NR<=$train_val_num)" $tmpdir/train_valid.scp > $tmpdir/cv.scp
awk "NR>$train_val_num" $tmpdir/train_valid.scp > $tmpdir/tt.scp

for x in tr cv tt; do
  ddir=${x}_synthetic
  mkdir -p ${data}/${ddir}
  cp $tmpdir/${x}.scp ${data}/${ddir}/wav.scp
  
  awk '{print($1, "dummy")}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

  awk '{print($1, "dummy")}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/text

  noise_wav_dir=${dns_wav}/noise/
  sed -e "s#${mixwav_dir}.*_\(.*\).wav#${noise_wav_dir}noise_fileid_\1.wav#g" ${data}/${ddir}/wav.scp \
    > ${data}/${ddir}/noise1.scp

  spk1_wav_dir=${dns_wav}/clean/
  sed -e "s#${mixwav_dir}.*_\(.*\).wav#${spk1_wav_dir}clean_fileid_\1.wav#g" ${data}/${ddir}/wav.scp \
    > ${data}/${ddir}/spk1.scp
done

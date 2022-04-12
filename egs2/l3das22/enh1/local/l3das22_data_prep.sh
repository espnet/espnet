#!/usr/bin/env bash

. ./path.sh



if [ $# -ne 1 ]; then
  echo "Arguments should be L3DAS wav path, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

L3DAS22=$1

tmpdir=data/temp
data=data
wavdata=data/L3DAS22_Multi

mkdir -p $tmpdir 


train_dir100=${L3DAS22}/L3DAS22_Task1_train100
train_dir360_1=${L3DAS22}/L3DAS22_Task1_train360_1
train_dir360_2=${L3DAS22}/L3DAS22_Task1_train360_2
dev_dir=${L3DAS22}/L3DAS22_Task1_dev

find $train_dir100 -name '*.txt' | sort -u > $tmpdir/tr100_txt.flist
find $train_dir360_1 -name '*.txt' | sort -u > $tmpdir/tr360_txt.flist
find $train_dir360_2 -name '*.txt' | sort -u >> $tmpdir/tr360_txt.flist
find $dev_dir -name '*.txt' | sort -u > $tmpdir/dev_txt.flist

for x in dev tr100 tr360; do
  # preparing transcript
  sed -e 's:.*/\(.*\).txt$:\1:i' $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.uttids
  while read line; do
      [ -f $line ] || error_exit "Cannot find transcription file '$line'";
      cat  "$line"  #| tr '\n' ' ' | perl -ape 's: *$:\n:;' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z  A-Z]//g'
  done < $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.trans
  paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_txt.trans \
  | sort -k1,1 > $tmpdir/${x}.trans

  # preparing clean speech
  sed -e 's:\(.*\).txt$:\1.wav:i' $tmpdir/${x}_txt.flist > $tmpdir/${x}_spk.flist
  paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_spk.flist \
  | sort -k1,1 > $tmpdir/${x}.spk
done

for x in dev tr; do
  if [ "$x" = "tr" ]; then
    ddir=train_multich
  elif [ "$x" = "dev" ]; then
    ddir=dev_multich
  fi

  wavdir=${wavdata}/L3DAS22_Task1_${x}*
  mkdir -p ${data}/${ddir}/single_channel

  for mic in A B; do
    for ch in {1..4}; do
      find $wavdir -name *${mic}_CH${ch}.wav  | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "_"); print(wav[1],$1)}' > ${data}/${ddir}/single_channel/wav_mic${mic}_ch${ch}.scp
    done
  done
  mix-mono-wav-scp.py ${data}/${ddir}/single_channel/wav_mic*_ch*.scp | sort -u> ${data}/${ddir}/wav.scp

  awk '{split($1, lst, "-"); spk=lst[1]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
    sort -u> ${data}/${ddir}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt
  
  if [ "$x" = "tr" ]; then
    cat $tmpdir/${x}100.trans $tmpdir/${x}360.trans | \
      sort -u >  ${data}/${ddir}/text

    cat $tmpdir/${x}100.spk $tmpdir/${x}360.spk | \
      sort -u>  ${data}/${ddir}/spk1.scp

  elif [ "$x" = "dev" ]; then
    cat $tmpdir/${x}.trans | \
      sort -u>  ${data}/${ddir}/text

    cat $tmpdir/${x}.spk | \
      sort -u>  ${data}/${ddir}/spk1.scp
  fi
done




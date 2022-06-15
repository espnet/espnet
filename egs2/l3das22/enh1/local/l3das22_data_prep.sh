#!/usr/bin/env bash

. ./path.sh

help_message=$(cat << EOF
Usage: $0 [L3DAS22 path] [wav path]
  required argument:
    L3DAS22 path: path to the L3DAS22 directory
    wav path: path to the splitted L3DAS22 wavfiles
    NOTE:
      You can download L3DAS22 manually from
        https://www.kaggle.com/datasets/l3dasteam/l3das22
EOF
)


if [ $# -ne 2 ]; then
  echo "${help_message}"
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

L3DAS22=$1
splitdir=$2

trap 'rm -rf "$tmpdir"' EXIT
tmpdir=$(mktemp -d /tmp/l3das22.XXXXXX)

data=data

train_dir100=$L3DAS22/L3DAS22_Task1_train100
train_dir360_1=$L3DAS22/L3DAS22_Task1_train360_1
train_dir360_2=$L3DAS22/L3DAS22_Task1_train360_2
dev_dir=$L3DAS22/L3DAS22_Task1_dev
test_dir=$L3DAS22/L3DAS22_Task1_test

find $train_dir100 -name '*.txt' | sort -u > $tmpdir/tr100_txt.flist
find $train_dir360_1 -name '*.txt' | sort -u > $tmpdir/tr360_txt.flist
find $train_dir360_2 -name '*.txt' | sort -u >> $tmpdir/tr360_txt.flist
find $dev_dir -name '*.txt' | sort -u > $tmpdir/dev_txt.flist
find $test_dir -name '*.txt' | sort -u > $tmpdir/test_txt.flist

for x in dev test tr100 tr360; do
  # preparing transcript
  sed -e 's:.*/\(.*\).txt$:\1:i' $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.uttids
  while read line; do
      [ -f $line ] || error_exit "Cannot find transcription file '$line'";
      cat  "$line"
  done < $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.trans
  paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_txt.trans \
  | sort -k1,1 > $tmpdir/${x}.trans

  # preparing clean speech
  sed -e 's:\(.*\).txt$:\1.wav:i' $tmpdir/${x}_txt.flist > $tmpdir/${x}_spk.flist
  paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_spk.flist \
  | sort -k1,1 > $tmpdir/${x}.spk
done

for x in dev test tr; do
  if [ "$x" = "tr" ]; then
    ddir=train
  else
    ddir=${x}
  fi

  split_data=$splitdir/L3DAS22_Task1_${x}*
  mkdir -p $data/${ddir}_multich
  mkdir -p $data/${ddir}_singlech
  mkdir -p $tmpdir/${ddir}_single_channel

  for mic in A B; do
    for ch in {1..4}; do
      find $split_data -name *${mic}_CH${ch}.wav  | sort -u | awk '{n=split($1, lst, "/"); split(lst[n], wav, "_"); print(wav[1],$1)}' > $tmpdir/${ddir}/wav_mic${mic}_ch${ch}.scp
    done
  done
  mix-mono-wav-scp.py $tmpdir/${ddir}/wav_mic*_ch*.scp | sort -u> $data/${ddir}_multich/wav.scp

  awk '{split($1, lst, "-"); spk=lst[1]; print($1, spk)}' $data/${ddir}_multich/wav.scp | \
    sort -u> $data/${ddir}_multich/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${ddir}_multich/utt2spk > $data/${ddir}_multich/spk2utt
  
  if [ "$x" = "tr" ]; then
    cat $tmpdir/${x}100.trans $tmpdir/${x}360.trans | \
      sort -u >  $data/${ddir}_multich/text

    cat $tmpdir/${x}100.spk $tmpdir/${x}360.spk | \
      sort -u>  $data/${ddir}_multich/spk1.scp

  elif [ "$x" = "dev" ] || [ "$x" = "test" ]; then
    cat $tmpdir/${x}.trans | \
      sort -u>  $data/${ddir}_multich/text

    cat $tmpdir/${x}.spk | \
      sort -u>  $data/${ddir}_multich/spk1.scp
  fi
  cp $data/${ddir}_multich/* $data/${ddir}_singlech
  cp $tmpdir/${ddir}_single_channel/wav_micA_ch1.scp $data/${ddir}_singlech/wav.scp
done




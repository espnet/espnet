#!/usr/bin/env bash

# Copyright 2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! $# -eq 2 ]; then
  echo "Arguments should be WSJ0-2MIX directory and the mixing script path, see ../run.sh for example."
  exit 1;
fi

. ./path.sh

wavdir=$1
srcdir=$2

# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# check if the script file exists.
for f in $srcdir/mix_2_spk_max_tr_mix $srcdir/mix_2_spk_max_cv_mix $srcdir/mix_2_spk_max_tt_mix; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done

rm -r data/{tr,cv,tt} 2>/dev/null

for x in tr cv tt; do
  mkdir -p data/$x
  cat $srcdir/mix_2_spk_max_${x}_mix | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > data/$x/wav.scp
done

awk '(ARGIND==1){spk[$1]=$2}(ARGIND==2){split($1, lst, "_"); spkr=spk[lst[3]]"_"spk[lst[5]]; print($1, spkr)}' data/wsj/train_si284/utt2spk data/tr/wav.scp | sort > data/tr/utt2spk
utt2spk_to_spk2utt.pl data/tr/utt2spk > data/tr/spk2utt
awk '(ARGIND==1){spk[$1]=$2}(ARGIND==2){split($1, lst, "_"); spkr=spk[lst[3]]"_"spk[lst[5]]; print($1, spkr)}' data/wsj/dev_dt_20/utt2spk data/cv/wav.scp | sort > data/cv/utt2spk
utt2spk_to_spk2utt.pl data/cv/utt2spk > data/cv/spk2utt
awk '(ARGIND==1){spk[$1]=$2}(ARGIND==2){split($1, lst, "_"); spkr=spk[lst[3]]"_"spk[lst[5]]; print($1, spkr)}' data/wsj/test_eval92/utt2spk data/tt/wav.scp | sort > data/tt/utt2spk
utt2spk_to_spk2utt.pl data/tt/utt2spk > data/tt/spk2utt

awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' data/wsj/train_si284/text data/tr/wav.scp | awk '{$2=""; print $0}' > data/tr/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' data/wsj/train_si284/text data/tr/wav.scp | awk '{$2=""; print $0}' > data/tr/text_spk2
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' data/wsj/dev_dt_20/text data/cv/wav.scp | awk '{$2=""; print $0}' > data/cv/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' data/wsj/dev_dt_20/text data/cv/wav.scp | awk '{$2=""; print $0}' > data/cv/text_spk2
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' data/wsj/test_eval92/text data/tt/wav.scp | awk '{$2=""; print $0}' > data/tt/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' data/wsj/test_eval92/text data/tt/wav.scp | awk '{$2=""; print $0}' > data/tt/text_spk2

echo "Mixture Data preparation succeeded"

#!/usr/bin/env bash
# Copyright 2021 Author: Yuekai Zhang
# Apache 2.0


. ./path.sh || exit 1;
# val or train
dataset=$1
corpus=$2
dst=$3
tmp=$dst/tmp
mkdir -p $tmp

echo "data_prep.sh: Preparing $dataset data in $corpus"


find $corpus/$dataset -name *.wav > $tmp/wav.flist   # attention directory arch
sed -e 's/\.wav//' $tmp/wav.flist | awk -F $dataset/ '{print $NF}' | sed -e 's/\//-/' > $tmp/utt.list
paste -d' ' $tmp/utt.list $tmp/wav.flist > $tmp/tmp_wav.scp

# remove first line wavfilename|size|transcript
sed 1d $corpus/${dataset}.csv > $tmp/text.1
# remove "|" and bytesize
awk -F\| '{$2="";print $0}' $tmp/text.1 > $tmp/text.2
# change first column from wav path to wav name, same as wav.scp
sed 's/\//-/1' $tmp/text.2 | sed -e 's/\.wav//' > $tmp/text.3

### normalized case
# remove all punctuation
awk '{for(x=2;x<=NF;x++){gsub(/[[:punct:]]/,"",$x)}}1' $tmp/text.3 > $tmp/text.4
# convert all upper letters to lower letters
tr '[:upper:]' '[:lower:]' <$tmp/text.4>$tmp/tmp_text

# wav.scp
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

# text
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_text | sort -k 1 | uniq > $tmp/text

# utt2spk & spk2utt
paste -d' ' $tmp/utt.list $tmp/utt.list > $tmp/tmp_utt2spk
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
utils/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt
# copy prepared resources from tmp_dir to target dir


### normalized case
for f in wav.scp text spk2utt utt2spk; do
  cp $tmp/$f $dst/$f || exit 1;
done
### unnormalized case
mkdir -p ${dst}_unnorm
for f in wav.scp spk2utt utt2spk; do
  cp $tmp/$f ${dst}_unnorm/$f || exit 1;
done
cp $tmp/text.3 ${dst}_unnorm/text

rm -r $tmp
echo "local/prepare_data.sh succeeded"
exit 0;

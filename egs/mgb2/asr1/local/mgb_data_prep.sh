#!/usr/bin/env bash

# Copyright (C) 2020 Kanari AI
# (Amir Hussein)

if [ $# -ne 4 ]; then
  echo "Usage: $0 <DB-dir> <process-xml> <data-subset> <mer>"
  exit 1;
fi

db_dir=$1
process_xml=$2
subset=$3  # subset of training data
mer=$4
test_dir=data/test

train_dir=data/train
dev_dir=data/dev

for x in $train_dir $dev_dir $test_dir; do
  mkdir -p $x
  if [ -f ${x}/wav.scp ]; then
    mkdir -p ${x}/.backup
    mv $x/{wav.scp,utt2spk,spk2utt,segments,text} ${x}/.backup
  fi
done

find $db_dir/train/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $train_dir/wav_list

# generate wav.scp file for training data
for x in $(cat $train_dir/wav_list); do
  echo $x $db_dir/train/wav/$x.wav >> $train_dir/wav.scp
done

# Creating subset of the train data for quick recipe testing
head -n $subset $train_dir/wav_list > $train_dir/wav_list.short

set -e -o pipefail

xmldir=$db_dir/train/xml/bw
if [ $process_xml == "python" ]; then
  echo "using python to process xml file"
  # check if bs4 and lxml are installin in python
  local/check_tools.sh
  # process xml file using python
  cat $train_dir/wav_list | while read basename; do
    [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
    local/process_xml.py $xmldir/$basename.xml - | local/add_to_datadir.py $basename $train_dir $mer
  done
elif [ $process_xml == 'xml' ]; then
  # check if xml binary exsits
  if command -v xml >/dev/null 2>/dev/null; then
    echo "using xml"
    cat $train_dir/wav_list | while read basename; do
      [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
      xml sel -t -m '//segments[@annotation_id="transcript_align"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename $train_dir
      echo $basename $wavDir/$basename.wav >> $train_dir/wav.scp
    done
  else
    echo "xml not found, you may use python by '--process-xml python'"
    exit 1;
  fi
else
  # invalid option
  echo "$0: invalid option for --process-xml, choose from 'xml' or 'python'"
  exit 1;
fi

# Generating necessary files for Dev
for x in text segments; do
  cp $db_dir/dev/${x}.all $dev_dir/${x}
done

find $db_dir/dev/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $dev_dir/wav_list

for x in $(cat $dev_dir/wav_list); do
  echo $x $db_dir/dev/wav/$x.wav >> $dev_dir/wav.scp
done

# Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" A"}' $dev_dir/wav.scp > $dev_dir/reco2file_and_channel

# Creating utt2spk for dev from segments
if [ ! -f $dev_dir/utt2spk ]; then
  cut -d ' ' -f1 $dev_dir/segments > $dev_dir/utt_id
  cut -d '_' -f1-2 $dev_dir/utt_id | paste -d ' ' $dev_dir/utt_id - > $dev_dir/utt2spk
fi
if [ ! -f $dev_dir/spk2utt ]; then
  utils/utt2spk_to_spk2utt.pl $dev_dir/utt2spk > $dev_dir/spk2utt
fi

# separate the overlapped dev files from non overlapped
for list in overlap non_overlap; do
  rm -rf ${dev_dir}_$list || true
  cp -r $dev_dir ${dev_dir}_$list
  for x in segments text utt2spk; do
    utils/filter_scp.pl $db_dir/dev/${list}_speech $dev_dir/$x > ${dev_dir}_$list/${x}
  done
done

# We are developing on non overapped data
cp ${dev_dir}_non_overlap/* $dev_dir


# Test
cp $db_dir/test/text $test_dir/text
cp $db_dir/test/segments.non_overlap_speech $test_dir/segments


find $db_dir/test/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $test_dir/wav_list

for x in $(cat $test_dir/wav_list); do
  echo $x $db_dir/test/wav/$x.wav >> $test_dir/wav.scp
done

# Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" A"}' $test_dir/wav.scp > $test_dir/reco2file_and_channel

# Creating utt2spk for test from segments
if [ ! -f $test_dir/utt2spk ]; then
  cut -d ' ' -f1 $test_dir/segments > $test_dir/utt_id
  cut -d '_' -f1-2 $test_dir/utt_id | paste -d ' ' $test_dir/utt_id - > $test_dir/utt2spk
fi
if [ ! -f $test_dir/spk2utt ]; then
  utils/utt2spk_to_spk2utt.pl $test_dir/utt2spk > $test_dir/spk2utt
fi

# separate the overlapped test files from non overlapped
for list in overlap non_overlap; do
  rm -rf ${test_dir}_$list || true
  cp -r $test_dir ${test_dir}_$list
  for x in segments text utt2spk; do
    utils/filter_scp.pl $db_dir/test/${list}_speech.lst $test_dir/$x > ${test_dir}_$list/${x}
  done
done

# We are testing on non overapped data
cp ${test_dir}_non_overlap/* $test_dir

# Train
find $db_dir/train/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $train_dir/wav_list

for x in $(cat $train_dir/wav_list); do
  echo $x $db_dir/train/wav/$x.wav >> $train_dir/wav.scp
done

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" A"}' $train_dir/wav.scp > $train_dir/reco2file_and_channel

if [ ! -f $train_dir/spk2utt ]; then
  utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt
fi

for dir in $train_dir $dev_dir ${dev_dir}_overlap ${dev_dir}_non_overlap $test_dir ${test_dir}_non_overlap; do
  utils/fix_data_dir.sh $dir
  utils/validate_data_dir.sh --no-feats $dir
done

for t in $train_dir $dev_dir ${dev_dir}_overlap ${dev_dir}_non_overlap; do

  sed -i 's/@@LAT@@//g' $t/text

done
# Subset of training data
train_subset=data/train_subset
mkdir -p $train_subset
utils/filter_scp.pl $train_dir/wav_list.short ${train_dir}/wav.scp > \
  ${train_subset}/wav.scp
cp ${train_dir}/{utt2spk,segments,spk2utt,text,reco2file_and_channel} ${train_subset}
utils/fix_data_dir.sh ${train_subset}

echo "Training and Test data preparation succeeded"

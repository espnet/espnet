#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

db_dir=${MGB2}
mer=80

 . utils/parse_options.sh || exit 1;

train_dir="data/train"
dev_dir="data/dev"
test_dir="data/eval"

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
while read line; do
  echo $line $db_dir/train/wav/$line.wav >> $train_dir/wav.scp
done < $train_dir/wav_list


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"
xmldir=$db_dir/train/xml/bw

log "using python to process xml file"
# check if bs4 and lxml are install in in python

if ! python3 -c "import bs4" 2>/dev/null; then
  log "$0: BeautifulSoup4 not installed, you can install it by 'pip install beautifulsoup4' if you prefer to use python to process xml file" 
  exit 1;
fi
if ! python3 -c "import lxml" 2>/dev/null; then
  log "$0: lxml not installed, you can install it by 'pip install lxml' if you prefer to use python to process xml file"
  exit 1;
fi

# process xml file using python
while read basename; do
  [ -e $xmldir/$basename.xml ] && local/process_xml.py $xmldir/$basename.xml - | local/add_to_datadir.py $basename $train_dir $mer
done < $train_dir/wav_list


# Generating necessary files for Dev 
for x in text segments; do
  cp $db_dir/dev/${x}.all $dev_dir/${x}
done

find $db_dir/dev/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $dev_dir/wav_list

while read line; do
  echo $line $db_dir/dev/wav/$line.wav >> $dev_dir/wav.scp
done < $dev_dir/wav_list

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

while read line; do
  echo $line $db_dir/test/wav/$line.wav >> $test_dir/wav.scp
done < $test_dir/wav_list

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

while read line; do
  echo $line $db_dir/train/wav/$line.wav >> $train_dir/wav.scp
done < $train_dir/wav_list

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" A"}' $train_dir/wav.scp > $train_dir/reco2file_and_channel

if [ ! -f $train_dir/spk2utt ]; then
  utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt
fi

for dir in $train_dir $dev_dir ${dev_dir}_overlap ${dev_dir}_non_overlap $test_dir ${test_dir}_non_overlap; do
  utils/fix_data_dir.sh $dir
  #utils/validate_data_dir.sh --no-feats $dir  # SOMETHING TO FIX HERE !!!!
done

for t in $train_dir $dev_dir ${dev_dir}_overlap ${dev_dir}_non_overlap; do

  sed -i 's/@@LAT@@//g' $t/text
  
done

log "Training and Test data preparation succeeded"
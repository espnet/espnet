#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <src-dir>"
  echo "e.g.: $0 data/dev"
  exit 1
fi

src=$1
set=`echo $src | awk -F"/" '{print $NF}'`

# fix the original directory at first
cp -rf data/$set data/$set.tmp
cut -f 1 -d " " data/$set/feats.scp > data/$set/reclist
reduce_data_dir.sh data/$set.tmp data/$set/reclist data/$set
if [ -f data/$set/text.en ]; then
    utils/fix_data_dir.sh --utt_extra_files "text.lc.es text.lc.en text.tc.es text.tc.en" data/$set
else
    utils/fix_data_dir.sh --utt_extra_files "text.lc.es text.lc.en.0 text.lc.en.1 text.lc.en.2 text.lc.en.3 \
                                             text.tc.es text.tc.en.0 text.tc.en.1 text.tc.en.2 text.tc.en.3" data/$set
fi
rm -rf data/$set.tmp


# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for Es
mkdir -p data/$set.es
for f in spk2utt utt2spk segments feats.scp wav.scp utt2num_frames spk2gender reco2file_and_channel; do
    sort data/$set/$f > data/$set.es/$f
done
sort data/$set/text.lc.es > data/$set.es/text
utils/fix_data_dir.sh data/$set.es || exit 1;
utils/validate_data_dir.sh data/$set.es || exit 1;


# for En
mkdir -p data/$set.en
for f in spk2utt utt2spk segments feats.scp wav.scp utt2num_frames spk2gender reco2file_and_channel; do
    sort data/$set/$f > data/$set.en/$f
done
if [ -f data/$set/text.lc.en ]; then
    sort data/$set/text.lc.en > data/$set.en/text
else
    sort data/$set/text.lc.en.0 > data/$set.en/text
    sort data/$set/text.lc.en.1 > data/$set.en/text.1
    sort data/$set/text.lc.en.2 > data/$set.en/text.2
    sort data/$set/text.lc.en.3 > data/$set.en/text.3
fi
utils/fix_data_dir.sh data/$set.en || exit 1;
utils/validate_data_dir.sh data/$set.en || exit 1;

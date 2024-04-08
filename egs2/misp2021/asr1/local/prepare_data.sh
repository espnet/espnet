#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Hang Chen, Yen-Ju Lu)
# Apache 2.0

# transform misp data to kaldi format

set -e -o pipefail

# configs
nj=1

. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1


if [ $# != 4 ]; then
  echo "Usage: $0 <corpus-data-dir> <enhancement-data-dir> <data-set> <store-dir>"
  echo " $0 /path/misp /path/misp_WPE train data/train_far"
  exit 1;
fi

data_root=$1
enhancement_root=$2
data_type=$3
store_dir=$4

# wav.scp segments text utt2spk
echo "prepare wav.scp segments text utt2spk"
python local/prepare_far_data.py -nj $nj $enhancement_root/audio $data_root/video $data_root/transcription $data_type $store_dir
cat $store_dir/temp/wav.scp | sort -k 1 | uniq > $store_dir/wav.scp
cat $store_dir/temp/mp4.scp | sort -k 1 | uniq > $store_dir/mp4.scp
cat $store_dir/temp/segments | sort -k 1 | uniq > $store_dir/segments
cat $store_dir/temp/utt2spk | sort -k 1 | uniq > $store_dir/utt2spk
cat $store_dir/temp/text_sentence | sort -k 1 | uniq > $store_dir/text

echo "prepare done"

# spk2utt
utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt

touch data/nlsyms.txt

echo "local/prepare_data.sh succeeded"
exit 0

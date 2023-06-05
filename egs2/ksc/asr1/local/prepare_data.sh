#!/usr/bin/env bash
# Copyright 2023 ISSAI (author: Yerbolat Khassanov)
# Apache 2.0

# To be run from one directory above this script.

. ./path.sh

KSC=$1

if [ -d "data" ]; then
  echo "data directory exists. skipping data preparation!"
  exit 0
fi

# Prepare: test, train, dev
for filename in $KSC/ISSAI_KSC_335RS_v1.1_flac/Meta/*; do
  dir=data/$(basename "$filename" .csv)
  rm -rf $dir && mkdir -p $dir
  echo $dir

  {
    read
    while IFS=" " read -r uttID others
    do
      echo "$uttID $(cat $KSC/ISSAI_KSC_335RS_v1.1_flac/Transcriptions/${uttID}.txt)" >> ${dir}/text
      echo "$uttID $KSC/ISSAI_KSC_335RS_v1.1_flac/Audios_flac/${uttID}.flac" >> ${dir}/wav.scp
      echo "$uttID $uttID" >> ${dir}/utt2spk
    done
  } < $filename

  cat $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt
  # Fix and check that data dirs are okay!
  utils/fix_data_dir.sh $dir
  utils/validate_data_dir.sh --no-feats $dir || exit 1
done

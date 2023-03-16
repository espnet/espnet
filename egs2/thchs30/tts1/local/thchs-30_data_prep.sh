#!/usr/bin/env bash
# Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.
#           2016  LeSpeech (Author: Xingyu Na)
#           2021  Carnegie Mellon University (Author: Jiatong Shi)

#This script pepares the data directory for thchs30 recipe.
#It reads the corpus and get wav.scp and transcriptions.

dir=$1
corpus_dir=$2


cd $dir

echo "creating data/{train,dev,test,train_phn,dev_phn,test_phn}"
mkdir -p data/{train,dev,test,train_phn,dev_phn,test_phn}

#create wav.scp, utt2spk.scp, spk2utt.scp, text
(
for x in train dev test; do
  echo "cleaning data/${x}"
  subset_dir=${dir}/data/${x}
  subset_phn_dir=${dir}/data/${x}_phn
  for f in wav.scp utt2spk spk2utt word.txt phone.txt text; do
      rm -rf "${subset_dir}/${f}"
  done
  echo "preparing scps and text in data/$x"
  for nn in `find  $corpus_dir/$x -name "*.wav" | sort -u | xargs -I {} basename {} .wav`; do
      spkid=`echo $nn | awk -F"_" '{print "" $1}'`
      spk_char=`echo $spkid | sed 's/\([A-Z]\).*/\1/'`
      spk_num=`echo $spkid | sed 's/[A-Z]\([0-9]\)/\1/'`
      spkid=$(printf '%s%.2d' "$spk_char" "$spk_num")
      utt_num=`echo $nn | awk -F"_" '{print $2}'`
      uttid=$(printf '%s%.2d_%.3d' "$spk_char" "$spk_num" "$utt_num")
      echo $uttid $corpus_dir/$x/$nn.wav >> "${subset_dir}"/wav.scp
      echo $uttid $spkid >> "${subset_dir}"/utt2spk
      echo $uttid `sed -n 1p $corpus_dir/data/$nn.wav.trn` >> "${subset_dir}"/word.txt
      echo $uttid `sed -n 3p $corpus_dir/data/$nn.wav.trn` >> "${subset_dir}"/phone.txt
  done

  cp "${subset_dir}"/word.txt "${subset_dir}"/text
  for f in wav.scp utt2spk text phone.txt; do
      sort "${subset_dir}"/${f} -o "${subset_dir}"/${f}
  done
  cp "${subset_dir}"/wav.scp "${subset_phn_dir}"/wav.scp
  cp "${subset_dir}"/phone.txt "${subset_phn_dir}"/text
  cp "${subset_dir}"/utt2spk "${subset_phn_dir}"/utt2spk
done
) || exit 1

for x in train dev test train_phn dev_phn test_phn; do
    utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    utils/fix_data_dir.sh data/${x}
done

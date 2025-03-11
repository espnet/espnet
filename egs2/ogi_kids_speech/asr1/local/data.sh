#!/bin/bash

# Set the data path
data_path=""

# Check the data path
if [ -z "$data_path" ]; then
    echo -e "ERROR: data_path is empty.\nThe dataset is OGI kids speech corpus, not publicly available so not provided here.\nIt is on https://catalog.ldc.upenn.edu/LDC2007S18\nPlease set the dataset path in /local/data.sh"
    exit 1
fi

# Define directories
wav_base_dir="$data_path/speech/scripted"
trans_file="$data_path/docs/all.map"
index_dir="$data_path/docs"
output_dir="data"

# Create the output directory and subdirectories if they don't exist
mkdir -p $output_dir/all $output_dir/train $output_dir/test $output_dir/train_dev $output_dir/train_nodev

# Initialize the Kaldi format files
for dir in all train test train_dev train_nodev; do
  : > "$output_dir/$dir/wav.scp"
  : > "$output_dir/$dir/text"
  : > "$output_dir/$dir/utt2spk"
done

# Create a dictionary for transcription mappings
trans_map_file="trans_map.txt"
: > "$trans_map_file"
while IFS= read -r line; do
  key=$(echo $line | cut -d' ' -f1 | tr '[:upper:]' '[:lower:]')
  value=$(echo $line | cut -d' ' -f2- | tr -d '"')
  echo "$key $value" >> $trans_map_file
done < $trans_file

# Process each index file
for index_file in "$index_dir"/*-verified.txt; do
  while IFS= read -r line; do
    wav_file=$(echo $line | cut -d' ' -f1 | sed "s#^../#$data_path/#")
    wav_path=$(realpath $wav_file)
    rel_path=${wav_path#"$wav_base_dir"/}

    spk_id=$(basename "$(dirname "$rel_path")")
    utt_id=$(basename $rel_path .wav)
    utt_key=${utt_id: -3:2}

    trans_text=$(grep "^$utt_key " $trans_map_file | head -n 1 | cut -d' ' -f2-)
    if [ -z "$trans_text" ]; then
      # Skip $utt_key due to missing transcription
      continue
    fi

    echo "$utt_id $wav_path" >> $output_dir/all/wav.scp
    echo "$utt_id $trans_text" >> $output_dir/all/text
    echo "$utt_id $spk_id" >> $output_dir/all/utt2spk
  done < $index_file
done

# Sort
for x in all test train train_dev train_nodev; do
  for f in text wav.scp utt2spk; do
    sort data/${x}/${f} -o data/${x}/${f}
  done
  utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
done

n_all=$(wc -l < data/all/text)
n_train=$((n_all * 4 / 5))
n_nodev=$((n_train * 3 / 4))
n_dev=$((n_train - n_nodev))
n_test=$((n_all - n_train))

utils/subset_data_dir.sh --first data/all "${n_train}" data/train
utils/subset_data_dir.sh --last data/all "${n_test}" data/test
utils/subset_data_dir.sh --first data/train "${n_dev}" data/train_dev
utils/subset_data_dir.sh --last data/train "${n_nodev}" data/train_nodev

rm $trans_map_file

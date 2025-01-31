#!/bin/bash

# Check if at least one argument is provided
if [ $# -ne 2 ]; then
    echo "ERROR: Please provide the OGI data path, and the output data path as arguments."
fi

data_path="$1"
output_dir="$2"

# Check the data path
if [ ! -d "$data_path" ]; then
    echo -e "ERROR: The specified data_path does not exist or is not a directory: $data_path"
    exit 1
fi

# Define directories
wav_base_dir="$data_path/speech/scripted"
trans_file="$data_path/docs/all.map"
index_dir="$data_path/docs"

# Create the output directory and subdirectories if they don't exist
mkdir -p $output_dir/all $output_dir/train $output_dir/dev $output_dir/test

# Initialize the Kaldi format files
for dir in all train dev test; do
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

# Find all WAV files recursively in the base directory
find "$wav_base_dir" -type f -name "*.wav" | while IFS= read -r wav_path; do
  rel_path=${wav_path#"$base_dir"/}
  
  spk_id=$(basename "$(dirname "$rel_path")")
  utt_id=$(basename "$rel_path" .wav)
  utt_key=${utt_id: -3:2}

  trans_text=$(grep "^$utt_key " "$trans_map_file" | head -n 1 | cut -d' ' -f2-)
  
  # Skip this utterance if transcription text is missing
  if [ -z "$trans_text" ]; then
      # Skip $utt_key due to missing transcription
      continue
  fi
  
  # Write to Kaldi output files
  echo "$utt_id $wav_path" >> "$output_dir/all/wav.scp"
  echo "$utt_id $trans_text" >> "$output_dir/all/text"
  echo "$utt_id $spk_id" >> "$output_dir/all/utt2spk"
done

# Sort
for f in text wav.scp utt2spk; do
  sort ${output_dir}/all/${f} -o ${output_dir}/all/${f}
done

utils/utt2spk_to_spk2utt.pl "$output_dir/all/utt2spk" > "$output_dir/all/spk2utt"

utils/subset_data_dir.sh --utt-list conf/file_list/train.list "$output_dir/all" "$output_dir/train"
utils/subset_data_dir.sh --utt-list conf/file_list/dev.list "$output_dir/all" "$output_dir/dev"
utils/subset_data_dir.sh --utt-list conf/file_list/test.list "$output_dir/all" "$output_dir/test"

rm $trans_map_file

echo "OGI scripted data preparation completed."

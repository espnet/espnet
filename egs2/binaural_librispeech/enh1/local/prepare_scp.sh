#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <clean_data_directory> <noisy_data_directory> <output_dir> <n_files>"
    exit 1
fi

# Assign command-line arguments to variables
clean_data_directory=$1
noisy_data_directory=$2
output_dir=$3
n_files=$4
output_clean_scp="${output_dir}/spk1.scp"
output_noisy_scp="${output_dir}/wav.scp"
output_utt2spk="${output_dir}/utt2spk"


if [ -e "$output_clean_scp" ] || [ -e "$output_noisy_scp" ] || [ -e "$output_utt2spk" ]; then
    echo "wav.scp, spk1.scp and utt2spk file already exists. Skipping..."
else
    # Create or clear the output SCP files and utt2spk file
    > "$output_clean_scp"
    > "$output_noisy_scp"
    > "$output_utt2spk"

    # Initialize counters
    found_files=0
    processed_files=0

    # Count total .flac files for progress tracking
    found_files=$(find $clean_data_directory -iname "*.flac" | wc -l)
    if [ ${found_files} -ne ${n_files} ]; then
        echo "Error: the total number of flac files (=${found_files}) does not match the expected number (${n_files}})."
        exit 1
    fi

    # Iterate over splits, speakers, chapters, and utterances
    for split in "$clean_data_directory"/*; do
        if [ -d "$split" ]; then
            for speaker in "$split"/*; do
                if [ -d "$speaker" ]; then
                    speaker_id=$(basename "$speaker")
                    for chapter in "$speaker"/*; do
                        if [ -d "$chapter" ]; then
                            for utterance in "$chapter"/*.flac; do
                                utterance_id=$(basename "$utterance")
                                clean_file_path="$utterance"
                                noisy_file_path="$noisy_data_directory/${split##*/}/${speaker##*/}/${chapter##*/}/$utterance_id"

                                # Write to the SCP files
                                echo "$utterance_id $clean_file_path" >> "$output_clean_scp"
                                echo "$utterance_id $noisy_file_path" >> "$output_noisy_scp"

                                # Write to the utt2spk file
                                echo "$utterance_id $speaker_id" >> "$output_utt2spk"

                                # Update progress
                                processed_files=$((processed_files + 1))
                                echo -ne "Progress: $processed_files/$found_files files processed\r"
                            done
                        fi
                    done
                fi
            done
        fi
    done
    
    if [ ${processed_files} -ne ${total_files} ]; then
        echo "Error: the total number of processed files (=${processed_files}) does not match the total number of files (=${total_files})."
        exit 1
    fi

    echo -e "\nSCP files created: $output_clean_scp, $output_noisy_scp"
    echo "utt2spk file created: $output_utt2spk"
fi

found_files=$(find $noisy_data_directory -iname "*.flac" | wc -l)
if [ ${found_files} -ne ${n_files} ]; then
    echo "Error: the total number of noisy files (=${found_files}) does not match the number of clean audio files (${total_files})."
    exit 1
fi
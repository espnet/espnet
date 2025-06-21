"""Prepares data for self-supervised audio only training with YT-8M."""

import glob
import os
import sys

DATA_READ_DIR = sys.argv[1]
DATA_WRITE_DIR = sys.argv[2]


def process_dataset(data_read_dir, data_write_dir):
    print(f"Processing YT-8M...", flush=True)

    # Find all mp3 files
    pattern = f"{data_read_dir}/**/*.mp3"
    mp3_files = glob.glob(pattern, recursive=True)
    print(len(mp3_files), "files found", flush=True)

    # Create output directory
    wav_scp_write_path = os.path.join(data_write_dir, "yt8m", "wav.scp")
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)

    processed_count = 0
    # Open the output file once
    with open(wav_scp_write_path, "w") as f:
        # Process each file sequentially
        for audio_path in mp3_files:
            if not os.path.exists(audio_path):
                continue

            filename = os.path.basename(audio_path)
            youtube_id = filename[:11]
            folder_name = os.path.basename(os.path.dirname(audio_path))  # category
            uttid = f"yt8m.{folder_name}.{youtube_id}"
            f.write(f"{uttid} {audio_path}\n")
            processed_count += 1

    print(f"Processed {processed_count} files.")
    return processed_count


if __name__ == "__main__":
    total_processed = process_dataset(DATA_READ_DIR, DATA_WRITE_DIR)
    print(f"Processed {total_processed} files.")

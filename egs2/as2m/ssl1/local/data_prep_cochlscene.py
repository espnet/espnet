"""Prepares data for self-supervised audio only training with CochlScene."""

import numpy as np
import soundfile as sf
import glob
import os
import sys
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATA_READ_DIR = sys.argv[1]
DATA_WRITE_DIR = sys.argv[2]
PARALLELISM = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SPLIT_THRESHOLD_SECONDS = 10
MIN_LENGTH_SECONDS = 0.2


def split_flac_into_chunks(audio_path: str, chunk_size: float):
    """
    Reads a FLAC file and splits it into chunks of a given size.
    """
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    audio_data, sample_rate = sf.read(audio_path)
    audio_duration = len(audio_data) / sample_rate

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    total_samples = int(sample_rate * audio_duration)
    chunk_samples = int(sample_rate * chunk_size)
    num_chunks = total_samples // chunk_samples

    chunks = {}
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = start_sample + chunk_samples
        chunk_length = (end_sample - start_sample) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            continue
        chunk_name = f"{filename}.chunk_{i+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:end_sample]

    if total_samples % chunk_samples != 0:
        start_sample = num_chunks * chunk_samples
        chunk_length = len(audio_data[start_sample:]) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            return chunks, sample_rate
        chunk_name = f"{filename}.chunk_{num_chunks+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:]

    return chunks, sample_rate


def process_file(audio_path, data_write_dir, wav_scp_f, wavscplock):
    """Processes a single file, handling splitting and writing metadata."""
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    uttid = f"cochlscene.{file_id}"

    if not os.path.exists(audio_path):
        return None

    audio_paths = []
    audio_chunks, sr = split_flac_into_chunks(audio_path, SPLIT_THRESHOLD_SECONDS)
    for chunk_name, audio in audio_chunks.items():
        chunk_path = os.path.join(data_write_dir, "cochlscene", "chunked", chunk_name)
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        sf.write(chunk_path, audio, sr)
        audio_paths.append(chunk_path)

    lines_to_write = [f"{uttid}.{i} {path}" for i, path in enumerate(audio_paths)]

    # Get the lock for the dataset and use it for writing
    with wavscplock:
        with open(wav_scp_f, "a") as f:
            f.write("\n".join(lines_to_write) + "\n")

    return len(lines_to_write)


def process_dataset(split_name, data_read_dir, data_write_dir, wavscplock):
    print(f"Processing CochlScene...")
    pattern = f"{data_read_dir}/{split_name}/**/*.wav"
    wav_files = glob.glob(pattern, recursive=True)
    wav_scp_write_path = os.path.join(data_write_dir, "cochlscene", "wav.scp")
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    processed_count = 0
    with ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda audio_path: process_file(
                        audio_path,
                        data_write_dir,
                        wav_scp_write_path,
                        wavscplock,
                    ),
                    wav_files,
                ),
                total=len(wav_files),
            )
        )
        processed_count = sum(1 for r in results if r is not None)
        created_files = sum(r for r in results if r is not None)

    print(f"Created {created_files} files from {split_name}.")
    print(f"Processed {processed_count} files from {split_name}.")
    return processed_count


if __name__ == "__main__":
    total_processed = 0
    for split in ["Train"]:
        dataset_lock = threading.Lock()
        total_processed += process_dataset(
            split, DATA_READ_DIR, DATA_WRITE_DIR, dataset_lock
        )
    print(f"Processed {total_processed} files in split {split}.")

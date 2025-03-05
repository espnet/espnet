"""Prepares data for SSL training with EPIC Sounds.
~ 150 hours of audio from the EPIC-Kitchens dataset.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

DATA_READ_FILE = sys.argv[1]
DATA_WRITE_DIR = sys.argv[2]
PARALLELISM = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SPLIT_THRESHOLD_SECONDS = 10
SAMPLE_RATE = 16000
MIN_LENGTH_SECONDS = 0.2


def split_hd5_audio_into_chunks(file_id: str, audio_data: np.array, chunk_size: float):
    """
    Splits audio into chunks of a given size.
    """
    sample_rate = SAMPLE_RATE

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    total_samples = len(audio_data)
    chunk_samples = int(sample_rate * chunk_size)
    num_chunks = total_samples // chunk_samples

    chunks = {}
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = start_sample + chunk_samples
        chunk_length = (end_sample - start_sample) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            continue
        chunk_name = f"{file_id}.chunk_{i+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:end_sample]

    if total_samples % chunk_samples != 0:
        start_sample = num_chunks * chunk_samples
        chunk_length = len(audio_data[start_sample:]) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            return chunks, sample_rate
        chunk_name = f"{file_id}.chunk_{num_chunks+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:]

    return chunks, sample_rate


def process_file(file_id, data, data_write_dir, wav_scp_f, wavscplock):
    """Processes a single file, handling splitting and writing metadata."""
    uttid = f"epic_sounds.{file_id}"

    audio_paths = []
    audio_chunks, sr = split_hd5_audio_into_chunks(
        file_id, data, SPLIT_THRESHOLD_SECONDS
    )
    for chunk_name, audio in audio_chunks.items():
        chunk_path = os.path.join(data_write_dir, "epic_sounds", "chunked", chunk_name)
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        sf.write(chunk_path, audio, sr)
        audio_paths.append(chunk_path)

    lines_to_write = [f"{uttid}.{i} {path}" for i, path in enumerate(audio_paths)]

    with wavscplock:
        with open(wav_scp_f, "a") as f:
            f.write("\n".join(lines_to_write) + "\n")
    return len(lines_to_write)


def process_dataset(data_read_file, data_write_dir, wavscplock):
    print(f"Processing epic sounds...")
    wav_scp_write_path = os.path.join(data_write_dir, "epic_sounds", "wav.scp")
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)

    processed_count = 0
    with ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        with h5py.File(data_read_file, "r") as dataset:
            results = list(
                tqdm(
                    executor.map(
                        lambda data_key: process_file(
                            data_key,
                            dataset[data_key],
                            data_write_dir,
                            wav_scp_write_path,
                            wavscplock,
                        ),
                        dataset,
                    ),
                    total=len(dataset),
                )
            )
        processed_count = sum(1 for r in results if r is not None)
        created_files = sum(r for r in results if r is not None)

    print(f"Created {created_files} files.")
    print(f"Processed {processed_count} files.")
    return processed_count


if __name__ == "__main__":
    total_processed = 0
    for split in ["train"]:
        dataset_lock = threading.Lock()
        total_processed += process_dataset(DATA_READ_FILE, DATA_WRITE_DIR, dataset_lock)
    print(f"Processed {total_processed} files in split {split}.")

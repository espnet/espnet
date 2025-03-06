"""Prepares data for self-supervised audio only training with FMA."""

import glob
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import soundfile as sf
import torchaudio
from tqdm import tqdm

DATA_READ_DIR = sys.argv[1]
DATA_WRITE_DIR = sys.argv[2]
PARALLELISM = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SPLIT_THRESHOLD_SECONDS = 10
MAX_SPLITS = 100
MIN_LENGTH_SECONDS = 0.2


def split_audio_into_chunks(audio_path: str, chunk_size: float):
    """
    Reads an audio file and splits it into chunks of a given size.
    """
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    # audio_data, sample_rate = sf.read(audio_path)
    metadata = torchaudio.info(audio_path)
    sample_rate = metadata.sample_rate
    num_frames = sample_rate * MAX_SPLITS * SPLIT_THRESHOLD_SECONDS
    audio_data, sample_rate = torchaudio.load(
        audio_path, num_frames=num_frames, channels_first=False
    )
    # audio_data, sample_rate = torchaudio.load(audio_path)

    audio_duration = audio_data.shape[0] / sample_rate
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(dim=1)

    total_samples = int(sample_rate * audio_duration)
    chunk_samples = int(sample_rate * chunk_size)
    num_chunks = total_samples // chunk_samples
    num_chunks = min(num_chunks, MAX_SPLITS)

    chunks = {}
    for i in range(num_chunks):
        # print("chunk", i, flush=True)
        start_sample = i * chunk_samples
        end_sample = start_sample + chunk_samples
        chunk_length = (end_sample - start_sample) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            continue
        chunk_name = f"{filename}.chunk_{i+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:end_sample].unsqueeze(-1)

    if total_samples % chunk_samples != 0:
        start_sample = num_chunks * chunk_samples
        end_sample = start_sample + chunk_samples
        chunk_length = len(audio_data[start_sample:end_sample]) / sample_rate
        if chunk_length < MIN_LENGTH_SECONDS:
            return chunks, sample_rate
        chunk_name = f"{filename}.chunk_{num_chunks+1}.flac"
        chunks[chunk_name] = audio_data[start_sample:end_sample].unsqueeze(-1)

    return chunks, sample_rate


def process_file(audio_path, data_write_dir, wav_scp_f, wavscplock):
    """Processes a single file, handling splitting and writing metadata."""
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    uttid = f"fma.{file_id}"

    if not os.path.exists(audio_path):
        return None

    audio_paths = []
    try:
        audio_chunks, sr = split_audio_into_chunks(audio_path, SPLIT_THRESHOLD_SECONDS)
        # print("split", file_id, flush=True)
        for chunk_name, audio in audio_chunks.items():
            chunk_path = os.path.join(data_write_dir, "fma", "chunked", chunk_name)
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            sf.write(chunk_path, audio.numpy(), sr)
            # torchaudio.save(chunk_path, audio, sr, format="wav", backend="ffmpeg")
            audio_paths.append(chunk_path)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

    if not audio_paths:
        return None

    lines_to_write = [f"{uttid}.{i} {path}" for i, path in enumerate(audio_paths)]

    # Get the lock for the dataset and use it for writing
    with wavscplock:
        with open(wav_scp_f, "a") as f:
            f.write("\n".join(lines_to_write) + "\n")
    # print("written to wav.scp", file_id, flush=True)
    return len(lines_to_write)


def process_dataset(data_read_dir, data_write_dir, wavscplock):
    print(f"Processing FMA...", flush=True)
    pattern = f"{data_read_dir}/**/*.mp3"
    mp3_files = glob.glob(pattern, recursive=True)
    print(len(mp3_files), "files found", flush=True)
    wav_scp_write_path = os.path.join(data_write_dir, "fma", "wav.scp")
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)

    processed_count = 0
    created_files = 0
    batch_size = 5000

    def batched(iterable, batch_size):
        it = iter(iterable)
        while batch := list(islice(it, batch_size)):
            yield batch

    print("Running with parallelism", PARALLELISM, flush=True)
    with ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        # for batch in batched(mp3_files, batch_size):
        # print("Processing batch...", flush=True)
        results = list(
            tqdm(
                executor.map(
                    lambda audio_path: process_file(
                        audio_path,
                        data_write_dir,
                        wav_scp_write_path,
                        wavscplock,
                    ),
                    mp3_files,
                ),
                total=len(mp3_files),
            )
        )
        processed_count += sum(1 for r in results if r is not None)
        created_files += sum(r for r in results if r is not None)

    print(f"Created {created_files} files.")
    print(f"Processed {processed_count} files.")
    return processed_count


if __name__ == "__main__":
    dataset_lock = threading.Lock()
    total_processed = process_dataset(DATA_READ_DIR, DATA_WRITE_DIR, dataset_lock)
    print(f"Processed {total_processed} files.")

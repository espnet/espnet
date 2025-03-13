#!/usr/bin/env python3
import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import multiprocessing
from functools import partial
import time
from multiprocessing import Lock
import sys


# Create a lock for file writing
file_lock = Lock()


def generate_unique_id(wav_path, file_count):
    """Generate a unique ID for a WAV file."""
    path_hash = str(abs(hash(os.path.abspath(wav_path))))[-8:]
    return f"{path_hash}_{file_count}"


def safe_load_audio(wav_path):
    """Load audio file with proper error handling."""
    try:
        y, sr = librosa.load(wav_path, sr=None)
        duration = len(y) / sr
        return y, sr, duration
    except Exception as e:
        print(f"Error loading audio file {wav_path}: {e}")
        raise


def write_metadata(
    utterance_id,
    start_time,
    end_time,
    wav_path,
    segment_file,
    wav_scp_file,
    utt2spk_file,
):
    """Write a segment entry to all metadata files with lock to prevent race conditions."""
    # Use lock to prevent multiple processes from writing simultaneously
    with file_lock:
        # Write to segment file
        with open(segment_file, "a") as f:
            if end_time == -1:
                end_time = start_time + 10.0
            f.write(f"{utterance_id} {utterance_id} {start_time:.2f} {end_time:.2f}\n")

        # Write to wav.scp file
        abs_wav_path = os.path.abspath(wav_path)
        with open(wav_scp_file, "a") as f:
            f.write(f"{utterance_id} {abs_wav_path}\n")

        # Write to utt2spk file
        with open(utt2spk_file, "a") as f:
            f.write(f"{utterance_id} {utterance_id}\n")


def create_chunks_with_noise(
    wav_path, mean_chunk_length=3.0, std_dev=0.5, target_length=10.0, noise_scale=0.01
):
    """Create audio chunks with noise padding to reach target length, using all original audio."""
    y, sr, total_duration = safe_load_audio(wav_path)

    # Calculate how many chunks we need based on mean chunk length
    chunks = []
    position = 0

    # Process all audio sequentially to ensure complete coverage
    while position < len(y):
        # Generate normally distributed chunk length around mean
        chunk_length = np.random.normal(mean_chunk_length, std_dev)
        chunk_length = max(1.0, min(chunk_length, target_length - 1))
        chunk_samples = int(chunk_length * sr)

        # Ensure we don't go past the end of the audio
        if position + chunk_samples > len(y):
            chunk_samples = len(y) - position
            chunk_length = chunk_samples / sr

        # Extract audio segment sequentially
        audio_segment = y[position : position + chunk_samples]
        position += chunk_samples  # Move position forward

        # Calculate padding needed to reach target length
        padding_length = target_length - chunk_length
        pre_pad = np.random.uniform(
            0.1, padding_length / 2
        )  # Limit pre-padding to half available space
        post_pad = padding_length - pre_pad
        pre_pad_samples = int(pre_pad * sr)
        post_pad_samples = int(post_pad * sr)

        # Generate noise for padding
        pre_noise = np.random.normal(0, noise_scale, pre_pad_samples)
        post_noise = np.random.normal(0, noise_scale, post_pad_samples)

        # Combine noise and audio
        combined = np.concatenate([pre_noise, audio_segment, post_noise])

        # Ensure exact target length
        target_samples = int(target_length * sr)
        if len(combined) > target_samples:
            combined = combined[:target_samples]
        elif len(combined) < target_samples:
            padding = np.random.normal(0, noise_scale, target_samples - len(combined))
            combined = np.concatenate([combined, padding])

        chunks.append((combined, sr))

    return chunks


def process_audio_file(args, wav_path, file_count):
    """Process a single WAV file. Returns 1 if successful, 0 if error."""
    try:
        # Unpack args
        segment_file, wav_scp_file, utt2spk_file, add_noise, noise_dir = args

        recording_id = generate_unique_id(wav_path, file_count)

        if not add_noise:
            # Regular chunking mode
            _, _, duration = safe_load_audio(wav_path)
            segment_duration = 10.0
            num_segments = int(duration // segment_duration)

            # Process complete segments
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                utterance_id = f"{recording_id}-{i:09d}"

                write_metadata(
                    utterance_id,
                    start_time,
                    end_time,
                    wav_path,
                    segment_file,
                    wav_scp_file,
                    utt2spk_file,
                )

            # Process remaining audio if any
            remaining_time = duration % segment_duration
            if remaining_time > 0:
                start_time = num_segments * segment_duration
                end_time = remaining_time + start_time
                utterance_id = f"{recording_id}-{num_segments:09d}"

                write_metadata(
                    utterance_id,
                    start_time,
                    end_time,
                    wav_path,
                    segment_file,
                    wav_scp_file,
                    utt2spk_file,
                )
        else:
            # Noise addition mode
            # Generate chunks with noise
            chunks = create_chunks_with_noise(wav_path)

            # Process each chunk
            for i, (chunk, sr) in enumerate(chunks):
                chunk_filename = f"{recording_id}-{i:04d}.wav"
                chunk_path = os.path.join(noise_dir, chunk_filename)

                # Save chunk to disk - needs lock since multiple processes might create files with same name
                with file_lock:
                    sf.write(chunk_path, chunk, sr)

                # Create metadata entries
                utterance_id = f"{recording_id}-{i:09d}"
                write_metadata(
                    utterance_id,
                    0.00,
                    10.00,
                    chunk_path,
                    segment_file,
                    wav_scp_file,
                    utt2spk_file,
                )
        return 1
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return 0


def initialize_output_files(write_dir, add_noise):
    """Create and initialize output files and directories."""
    os.makedirs(write_dir, exist_ok=True)

    segment_file = os.path.join(write_dir, "segments")
    wav_scp_file = os.path.join(write_dir, "wav.scp")
    utt2spk_file = os.path.join(write_dir, "utt2spk")

    # Create noise directory if needed
    noise_dir = None
    if add_noise:
        noise_dir = os.path.join(write_dir, "noise_chunks")
        os.makedirs(noise_dir, exist_ok=True)

    # Create empty output files
    for file_path in [segment_file, wav_scp_file, utt2spk_file]:
        open(file_path, "w").close()

    return segment_file, wav_scp_file, utt2spk_file, noise_dir


def process_files_parallel(
    read_dir,
    segment_file,
    wav_scp_file,
    utt2spk_file,
    add_noise=False,
    num_processes=None,
):
    """Process all WAV files in the given directory in parallel."""
    # First gather all wav files paths
    wav_files = []
    for root, _, files in os.walk(read_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    # If no wav files found
    if not wav_files:
        print("No WAV files found in the specified directory.")
        return 0

    # Create noise directory if needed
    noise_dir = None
    if add_noise:
        noise_dir = os.path.join(os.path.dirname(segment_file), "noise_chunks")
        os.makedirs(noise_dir, exist_ok=True)

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    # Create a pool of workers
    print(f"Processing {len(wav_files)} WAV files using {num_processes} processes...")

    # Prepare arguments for the worker function
    args = (segment_file, wav_scp_file, utt2spk_file, add_noise, noise_dir)

    # Set up progress tracking for parallel execution
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap to process files and tqdm to track progress
        # starmap unpacks the arguments correctly unlike imap
        file_id_pairs = [(wav_files[i], i + 1) for i in range(len(wav_files))]
        worker_func = partial(process_audio_file, args)

        results = list(
            tqdm(
                pool.starmap(
                    worker_func,
                    file_id_pairs,
                    chunksize=max(1, len(wav_files) // (num_processes * 10)),
                ),
                total=len(wav_files),
                desc="Processing audio files",
                unit="file",
                file=sys.stdout,
                mininterval=1.0,
            )
        )

    # Count successful and failed files
    processed_files = sum(results)
    error_files = len(wav_files) - processed_files

    if error_files > 0:
        print(f"Skipped {error_files} corrupted or invalid files")

    return processed_files


def main():
    parser = argparse.ArgumentParser(
        description="Create segments from WAV files with parallel processing"
    )
    parser.add_argument("READ_DIR", help="Directory to read WAV files from")
    parser.add_argument("WRITE_DIR", help="Directory to write output files to")
    parser.add_argument(
        "--add_noise", action="store_true", help="Add noise to the audio"
    )
    parser.add_argument(
        "--chunk_length",
        type=float,
        default=3.0,
        help="Mean chunk length for noise segments",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.01,
        help="Noise intensity scale (0.01-0.1)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: CPU count - 1)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Initialize output files
    segment_file, wav_scp_file, utt2spk_file, noise_dir = initialize_output_files(
        args.WRITE_DIR, args.add_noise
    )

    # Process all WAV files in parallel
    processed_files = process_files_parallel(
        args.READ_DIR,
        segment_file,
        wav_scp_file,
        utt2spk_file,
        args.add_noise,
        args.processes,
    )

    elapsed_time = time.time() - start_time
    print(
        f"Successfully processed {processed_files} WAV files in {elapsed_time:.2f} seconds "
        f"({processed_files/elapsed_time:.2f} files/sec) and created output files in {args.WRITE_DIR}"
    )


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()

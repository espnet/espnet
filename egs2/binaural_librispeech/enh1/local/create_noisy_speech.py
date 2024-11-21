import os
import random
import sys
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Function to add noise to an audio file
def add_noise(audio, noise, snr_db):
    """
    Add noise to the audio with a specified signal-to-noise ratio (SNR).
    Args:
        audio (np.array): The original audio signal.
        noise (np.array): The noise signal.
        snr_db (float): Desired signal-to-noise ratio in decibels.
    Returns:
        np.array: The noisy audio signal.
    """
    # Calculate RMS (Root Mean Square) of both audio and noise
    rms_audio = np.sqrt(np.mean(audio ** 2))
    rms_noise = np.sqrt(np.mean(noise ** 2))

    # Adjust noise level to match the desired SNR
    snr = 10 ** (snr_db / 20)  # Convert dB to linear scale
    scaled_noise = noise * (rms_audio / (snr * rms_noise))

    # Ensure noise and audio have the same length
    len_audio = audio.shape[-1]
    if len(noise) < len_audio:
        noise = np.tile(scaled_noise, int(np.ceil(len_audio / len(scaled_noise))))[:len_audio]
    else:
        noise = scaled_noise[:len_audio]

    # Add noise to the audio signal
    noisy_audio = audio + noise
    return noisy_audio

# Function to process a single audio file
def process_audio_file(args):
    audio_path, noise_file, output_dir, audio_dir = args

    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if sr <16000:
        print(f"read file with {audio_path=} with {sr=}, skipping...")
        return

    noise, _ = librosa.load(noise_file, sr=sr)

    # Generate a random SNR between -10 and 20
    snr_db = random.uniform(-10, 20)

    # Add noise to the audio
    noisy_audio = add_noise(audio, noise, snr_db)

    # Define output path and save the noisy audio
    relative_path = os.path.relpath(os.path.dirname(audio_path), audio_dir)
    output_path = os.path.join(output_dir, relative_path)

    # Ensure the subdirectory structure is maintained
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, os.path.basename(audio_path))
    sf.write(output_file, noisy_audio.T, sr)


# Function to process audio files
def process_audio_files(audio_dir, noise_dir, output_dir, n_jobs):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    noise_files = []
    for root, _, files in os.walk(noise_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                noise_files.append(os.path.join(root, file))

    # Collect all audio files and crate args for function
    args = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.flac'):  # Extend with other audio formats if needed
                noise_file = random.choice(noise_files)
                args.append((os.path.join(root, file), noise_file, output_dir, audio_dir))

    # Use multiprocessing to process audio files
    pool = Pool(processes=n_jobs)

    for _ in tqdm(pool.imap(func=process_audio_file, iterable=args), total=len(args)):
        pass

    pool.close()
    pool.join()

    print("Finished generating noisy speech.")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python create_noisy_speech.py <audio_dir> <noise_dir> <output_dir> <n_jobs>")
        sys.exit(1)

    # Get directories from command-line arguments
    audio_dir = sys.argv[1]
    noise_dir = sys.argv[2]
    output_dir = sys.argv[3]
    n_jobs = int(sys.argv[4])

    # Process the audio files
    process_audio_files(audio_dir, noise_dir, output_dir, n_jobs)

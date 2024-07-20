# simulation script taken from https://github.com/sp-uhh/sgmse

import argparse
import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
from librosa import load
from librosa.core import resample
from soundfile import write
from tqdm import tqdm

# Python script for generating noisy mixtures for training
#
# Mix WSJ0 with CHiME3 noise with SNR sampled uniformly in [min_snr, max_snr]


min_snr = 0
max_snr = 20
sr = 16000


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("wsj0", type=str, help="path to WSJ0 directory")
    parser.add_argument("chime3", type=str, help="path to CHiME3 directory")
    parser.add_argument("target", type=str, help="target path for training files")
    args = parser.parse_args()

    # Clean speech for training
    train_speech_files = sorted(glob(args.wsj0 + "**/si_tr_s/**/*.wav", recursive=True))
    valid_speech_files = sorted(
        glob(args.wsj0 + "**/si_dt_05/**/*.wav", recursive=True)
    )
    test_speech_files = sorted(glob(args.wsj0 + "**/si_et_05/**/*.wav", recursive=True))

    noise_files = glob(args.chime3 + "/**/backgrounds/*.wav", recursive=True)

    noise_files = [file for file in noise_files if (file[-7:-4] == "CH1")]

    # Load CHiME3 noise files
    noises = []
    print("Loading CHiME3 noise files")
    for file in noise_files:
        noise = load(file, sr=None)[0]
        noises.append(noise)

    # Create target dir
    train_clean_path = Path(os.path.join(args.target, "train/clean"))
    train_noisy_path = Path(os.path.join(args.target, "train/noisy"))
    valid_clean_path = Path(os.path.join(args.target, "valid/clean"))
    valid_noisy_path = Path(os.path.join(args.target, "valid/noisy"))
    test_clean_path = Path(os.path.join(args.target, "test/clean"))
    test_noisy_path = Path(os.path.join(args.target, "test/noisy"))

    train_clean_path.mkdir(parents=True, exist_ok=True)
    train_noisy_path.mkdir(parents=True, exist_ok=True)
    valid_clean_path.mkdir(parents=True, exist_ok=True)
    valid_noisy_path.mkdir(parents=True, exist_ok=True)
    test_clean_path.mkdir(parents=True, exist_ok=True)
    test_noisy_path.mkdir(parents=True, exist_ok=True)

    # Initialize seed for reproducability
    np.random.seed(0)

    # Create files for training
    print("Create training files")
    for i, speech_file in enumerate(tqdm(train_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_ind = np.random.randint(len(noises))
        speech_power = 1 / len(s) * np.sum(s**2)

        n = noises[noise_ind]
        start = np.random.randint(len(n) - len(s))
        n = n[start : start + len(s)]

        noise_power = 1 / len(n) * np.sum(n**2)
        noise_power_target = speech_power * np.power(10, -snr_dB / 10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split("/")[-1]
        write(os.path.join(train_clean_path, file_name), s, sr)
        write(os.path.join(train_noisy_path, file_name), x, sr)

    # Create files for validation
    print("Create validation files")
    for i, speech_file in enumerate(tqdm(valid_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_ind = np.random.randint(len(noises))
        speech_power = 1 / len(s) * np.sum(s**2)

        n = noises[noise_ind]
        start = np.random.randint(len(n) - len(s))
        n = n[start : start + len(s)]

        noise_power = 1 / len(n) * np.sum(n**2)
        noise_power_target = speech_power * np.power(10, -snr_dB / 10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split("/")[-1]
        write(os.path.join(valid_clean_path, file_name), s, sr)
        write(os.path.join(valid_noisy_path, file_name), x, sr)

    # Create files for test
    print("Create test files")
    for i, speech_file in enumerate(tqdm(test_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_ind = np.random.randint(len(noises))
        speech_power = 1 / len(s) * np.sum(s**2)

        n = noises[noise_ind]
        start = np.random.randint(len(n) - len(s))
        n = n[start : start + len(s)]

        noise_power = 1 / len(n) * np.sum(n**2)
        noise_power_target = speech_power * np.power(10, -snr_dB / 10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split("/")[-1]
        write(os.path.join(test_clean_path, file_name), s, sr)
        write(os.path.join(test_noisy_path, file_name), x, sr)

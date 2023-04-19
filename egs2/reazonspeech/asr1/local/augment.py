#!/usr/bin/env python3

import getopt
import glob
import io
import os
import random
import sys

import librosa
import numpy as np
import soundfile

SAMPLERATE = 16000


def encode_audio(audio):
    tmp = io.BytesIO()
    soundfile.write(tmp, audio, SAMPLERATE, format="flac")
    return tmp.getvalue()


def apply_cutoff(audio):
    if len(audio) < SAMPLERATE * 4:
        return audio

    lower = int(min(len(audio) / 4, SAMPLERATE * 0.2))
    upper = int(SAMPLERATE * 2)
    if upper - lower > 0.3 * len(audio):
        return audio
    return audio[random.randint(lower, upper) :]


def apply_musan(audio, musan_dir):
    # Get a random noise
    files = glob.glob(os.path.join(musan_dir, "noise/*/*.wav"))
    noise = librosa.load(random.choice(files), sr=SAMPLERATE)[0]

    # Calculate signal/noise power
    signal_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(noise))

    # Convert SNR(db) into scaling factor (adjust loudness according to SNR)
    snr = random.uniform(-9, 9)
    snr_ratio = np.power(10, -snr / 10)
    multiplier = np.sqrt(snr_ratio * signal_power / noise_power)

    # Repeat the noise up to the input audio length
    noise = librosa.util.fix_length(noise, size=len(audio), mode="wrap")
    return audio + multiplier * noise


def main():
    cutoff = False
    musan_dir = None

    opts, args = getopt.getopt(sys.argv[1:], "cm:")
    for k, v in opts:
        if k == "-c":
            cutoff = True
        elif k == "-m":
            musan_dir = v

    # Read audio
    buf = []
    for audio_filepath in args:
        buf.append(librosa.load(audio_filepath, sr=SAMPLERATE)[0])
    audio = np.concatenate(buf)

    # Data Augmentation
    if cutoff:
        audio = apply_cutoff(audio)
    if musan_dir:
        audio = apply_musan(audio, musan_dir)

    sys.stdout.buffer.write(encode_audio(audio))


if __name__ == "__main__":
    main()

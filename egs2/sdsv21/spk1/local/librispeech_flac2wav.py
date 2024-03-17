# librispeech_flac2wav.py

#  uses ffmpeg to convert LibriSpeech .flac audio files to .wav

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def convert_to_wav(src_path, dst_path):
    command = ["ffmpeg", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def process_file(file_info):
    src_path, dst_path, dst_dir = file_info
    os.makedirs(dst_dir, exist_ok=True)
    convert_to_wav(src_path, dst_path)


def main(args):
    src = args.src
    dst = args.dst
    n_proc = args.n_proc
    files_to_process = []

    for root, directories, files in os.walk(src):
        for file in files:
            if os.path.splitext(file)[1] == ".flac":
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, src)
                dst_dir = os.path.join(dst, relative_path)
                wav_file_name = os.path.splitext(file)[0] + ".wav"
                dst_path = os.path.join(dst_dir, wav_file_name)
                files_to_process.append((src_path, dst_path, dst_dir))

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        list(
            tqdm(
                executor.map(process_file, files_to_process),
                total=len(files_to_process),
            )
        )

    print("Conversion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLAC to WAV Converter")
    parser.add_argument("--src", type=str, required=True, help="Src dir of FLAC files")
    parser.add_argument("--dst", type=str, required=True, help="Dst dir for WAV files")
    parser.add_argument(
        "--n_proc", type=int, default=os.cpu_count(), help="N proc to use"
    )
    args = parser.parse_args()

    main(args)

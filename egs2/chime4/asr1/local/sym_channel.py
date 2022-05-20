import argparse
import os
from os import path


def create_sym(data_dir, track, wav):
    ori_path = path.join(f"{data_dir}/data/audio/16kHz/isolated", wav)
    wav_path = path.join(
        f"{data_dir}/data/audio/16kHz/isolated_{track}_track", *wav.split("/")[:-1]
    )
    if not path.exists(wav_path):
        os.makedirs(wav_path)
    if track == "1ch":
        new_wav = wav.split("/")[-1].split(".")[0] + ".wav"
        os.system(" ".join(["ln -s", ori_path, path.join(wav_path, new_wav)]))
    elif track == "2ch":
        os.system(" ".join(["ln -s", ori_path, wav_path]))


def create_sym_list(data_dir, track):
    for root, dirs, files in os.walk(f"{data_dir}/data/annotations"):
        for file in files:
            list_file = path.join(root, file)
            if ".list" not in list_file or track not in list_file:
                continue
            with open(list_file, "r") as lfile:
                for line in lfile.readlines():
                    for wav in line.split():
                        create_sym(data_dir, track, wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("data_dir", type=str, default="wpe", help="dir of misp data")
    parser.add_argument("track", type=str, default="wpe", help="1ch or 2ch")
    args = parser.parse_args()

    create_sym_list(args.data_dir, args.track)

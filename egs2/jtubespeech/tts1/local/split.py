#!/usr/bin/env python3

# Copyright 2021 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import os

import soundfile as sf
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_raw", type=str, default="downloads/jtuberaw", required=False
    )
    parser.add_argument(
        "--db_split", type=str, default="downloads/jtubesplit", required=False
    )
    args = parser.parse_args()

    rawdata_dir = os.path.join(os.path.dirname(__file__), "../{}".format(args.db_raw))
    outdata_dir = os.path.join(os.path.dirname(__file__), "../{}".format(args.db_split))
    text_paths = glob.glob(os.path.join(rawdata_dir, "txt", "*", "*.txt"))

    os.makedirs(outdata_dir, exist_ok=True)
    os.makedirs(os.path.join(outdata_dir, "wav16k"), exist_ok=True)
    transcript_path = os.path.join(outdata_dir, "transcript_raw.txt")
    if os.path.exists(transcript_path):
        os.remove(transcript_path)

    for path in text_paths:
        dirname = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(os.path.join(outdata_dir, "wav16k", stem), exist_ok=True)
        wav, sr = sf.read(
            os.path.join(rawdata_dir, "wav16k", dirname, "{}.wav".format(stem))
        )
        with open(path, "r") as fr:
            for i, line in enumerate(fr):
                line_list = line.strip().split("\t", 2)
                idx = "0" * (4 - len(str(i))) + str(i)
                t_s, t_e, transcript = (
                    float(line_list[0]),
                    float(line_list[1]),
                    line_list[2].strip('"'),
                )
                wav_seg = wav[int(sr * t_s) : int(sr * t_e)]
                stem_seg = stem + "_" + idx
                sf.write(
                    os.path.join(
                        outdata_dir, "wav16k", stem, "{}.wav".format(stem_seg)
                    ),
                    wav_seg,
                    sr,
                )
                with open(transcript_path, "a") as fa:
                    fa.write("{} {}\n".format(stem_seg, transcript))

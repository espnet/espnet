# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This file is from Libri-Light repo: https://github.com/facebookresearch/
# libri-light/blob/main/data_preparation/cut_by_vad.py
import argparse
import json
import multiprocessing
import pathlib

import numpy as np
import soundfile as sf
import tqdm


def save(seq, fname, index, extension):
    output = np.hstack(seq)
    file_name = fname.parent / (fname.stem + f"_{index:04}{extension}")
    fname.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, samplerate=16000)


def cut_sequence(path, vad, path_out, target_len_sec, out_extension):
    data, samplerate = sf.read(path)

    assert len(data.shape) == 1
    assert samplerate == 16000

    to_stitch = []
    length_accumulated = 0.0

    i = 0
    for start, end in vad:
        start_index = int(start * samplerate)
        end_index = int(end * samplerate)
        slice = data[start_index:end_index]

        # if a slice is longer than target_len_sec, we put it entirely in it's own piece
        if (
            length_accumulated + (end - start) > target_len_sec
            and length_accumulated > 0
        ):
            save(to_stitch, path_out, i, out_extension)
            to_stitch = []
            i += 1
            length_accumulated = 0

        to_stitch.append(slice)
        length_accumulated += end - start

    if to_stitch:
        save(to_stitch, path_out, i, out_extension)


def cut_book(task):
    path_book, root_out, target_len_sec, extension = task

    speaker = pathlib.Path(path_book.parent.name)

    for i, meta_file_path in enumerate(path_book.glob("*.json")):
        with open(meta_file_path, "r") as f:
            meta = json.loads(f.read())
        book_id = meta["book_meta"]["id"]
        vad = meta["voice_activity"]

        sound_file = meta_file_path.parent / (meta_file_path.stem + ".flac")

        path_out = root_out / speaker / book_id / (meta_file_path.stem)
        cut_sequence(sound_file, vad, path_out, target_len_sec, extension)


def cut(input_dir, output_dir, target_len_sec=30, n_process=32, out_extension=".flac"):
    list_dir = pathlib.Path(input_dir).glob("*/*")
    list_dir = [x for x in list_dir if x.is_dir()]

    print(f"{len(list_dir)} directories detected")
    print(f"Launching {n_process} processes")

    tasks = [
        (path_book, output_dir, target_len_sec, out_extension) for path_book in list_dir
    ]

    with multiprocessing.Pool(processes=n_process) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(cut_book, tasks), total=len(tasks)):
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut a dataset in small " "sequences using VAD files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to the input directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory",
        required=True,
    )

    parser.add_argument(
        "--target_len_sec",
        type=int,
        default=60,
        help="Target time, in seconds of each output sequence" "(default is 60)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=32, help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--out_extension",
        type=str,
        default=".flac",
        choices=[".wav", ".flac", ".mp3"],
        help="Output extension",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    cut(
        args.input_dir,
        args.output_dir,
        args.target_len_sec,
        args.n_workers,
        args.out_extension,
    )

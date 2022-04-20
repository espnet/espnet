#!/usr/bin/env python

# Copyright 2022  Shanghai Jiao Tong University (Author: Wangyou Zhang)
# Apache 2.0
from pathlib import Path
import re
import sys


def prepare_audioset_category(audio_list, audioset_dir, output_file, skip_csv_rows=3):
    audios = []
    with Path(audio_list).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            audios.append(line.strip())

    utt2category = {}
    for csv in Path(audioset_dir).rglob("*.csv"):
        with csv.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < skip_csv_rows:
                    continue
                # --PJHxphWEs, 30.000, 40.000, "/m/09x0r,/t/dd00088"
                try:
                    YTID, start_seconds, end_seconds, positive_labels = re.split(
                        r",\s*", line.strip(), maxsplit=3
                    )
                except ValueError as err:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    err_file = Path(exc_tb.tb_frame.f_code.co_filename).name
                    err_file_line = exc_tb.tb_lineno
                    print(
                        "=== Warning: skipping '%s' due to the following error ==="
                        % csv
                    )
                    print(
                        "  [%s (line %s)] %s: %s"
                        % (err_file, err_file_line, exc_type.__name__, err)
                    )
                    break
                positive_labels = re.sub(r'^"(.*)"$', r"\1", positive_labels)
                positive_labels = ",".join(sorted(re.split(r",\s*", positive_labels)))
                utt2category[YTID] = positive_labels

    ret = []
    for audio in audios:
        ytid = re.sub(r"(.*)_\d+\.\d+", r"\1", Path(audio).stem)
        ret.append("%s %s\n" % (ytid, utt2category[ytid]))

    outfile = Path(output_file)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        for line in ret:
            f.write(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_list",
        type=str,
        help="Path to a text file containing the list of audios in Audioset",
    )
    parser.add_argument(
        "--audioset_dir",
        type=str,
        required=True,
        help="Path to the Audioset root directory",
    )
    parser.add_argument(
        "--skip_csv_rows",
        type=str,
        default=3,
        help="Line numbers to skip from top while reading csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the file for write audio list with category information",
    )
    args = parser.parse_args()

    prepare_audioset_category(
        args.audio_list,
        args.audioset_dir,
        args.output_file,
        skip_csv_rows=args.skip_csv_rows,
    )

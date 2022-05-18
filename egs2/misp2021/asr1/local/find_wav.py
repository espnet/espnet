#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import glob
import os


def find_wav(data_root, scp_dir, scp_name="wpe", wav_type="Far", n_split=1):
    type2group_num = {"Far": 6, "Middle": 2}
    wav_dir = os.path.join(data_root, wav_type.lower())
    all_wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))

    sorted_wav_paths = sorted(all_wav_paths)
    lines = ["" for _ in range(n_split)]
    for wav_idx in range(len(sorted_wav_paths) // type2group_num[wav_type]):
        line = sorted_wav_paths[wav_idx * type2group_num[wav_type]]
        group_name = "_".join(line.split("/")[-1].split("_")[:5])
        line = group_name + " " + line
        for i in range(1, type2group_num[wav_type]):
            line = line + " " + sorted_wav_paths[wav_idx * type2group_num[wav_type] + i]
        line += "\n"
        lines[wav_idx % n_split] += line

    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir, exist_ok=True)

    if n_split == 1:
        with codecs.open(
            os.path.join(scp_dir, "{}.scp".format(scp_name)), "w"
        ) as handle:
            handle.write(lines[0])
        return None
    for j in range(n_split):
        with codecs.open(
            os.path.join(scp_dir, "{}.{}.scp".format(scp_name, j + 1)), "w"
        ) as handle:
            handle.write(lines[j])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("data_dir", type=str, default="wpe", help="dir of misp data")
    parser.add_argument("scp_dir", type=str, default="wpe", help="dir of scp file")
    parser.add_argument("scp_name", type=str, default="wpe", help="name of scp file")
    parser.add_argument("wav_type", type=str, default="Far", help="wav type")
    parser.add_argument("-nj", type=int, default=1, help="number of split files")
    args = parser.parse_args()

    find_wav(
        data_root=args.data_dir,
        scp_dir=args.scp_dir,
        scp_name=args.scp_name,
        wav_type=args.wav_type,
        n_split=args.nj,
    )

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import tqdm


def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def glob_check(root_folder, has_eval=False, input_json=None):
    all_files = []
    for ext in [".json", ".uem", ".wav", ".flac"]:
        all_files.extend(
            glob.glob(os.path.join(root_folder, "**/*{}".format(ext)), recursive=True)
        )

    for f in tqdm.tqdm(all_files):
        digest = md5_file(f)
        if not has_eval and Path(f).parent == "eval":
            continue

        if not input_json[str(Path(f).relative_to(root_folder))] == digest:
            raise RuntimeError(
                "MD5 Checksum for {} is not the same. "
                "Data has not been generated correctly."
                "You can retry to generate it or re-download it."
                "If this does not work, please reach us. ".format(
                    str(Path(f).relative_to(root_folder))
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute MD5 hash for each file recursively to check"
        "if the data generation and download was successful or not."
    )

    parser.add_argument(
        "-c,--chime7dasr_root",
        type=str,
        metavar="STR",
        dest="chime7_root",
        help="Path to chime7dasr dataset main directory."
        "It should contain chime6, dipco and mixer6 as sub-folders.",
    )
    parser.add_argument(
        "-e,--has_eval",
        required=False,
        type=int,
        default=0,
        dest="has_eval",
        help="Whether to check also " "for evaluation (released later).",
    )
    parser.add_argument(
        "-i,--input_json",
        type=str,
        default="./local/chime7_dasr_md5.json",
        dest="input_json",
        required=False,
        help="Input JSON file to check against containing md5 checksums for each file.",
    )
    args = parser.parse_args()
    with open(args.input_json, "r") as f:
        checksum_json = json.load(f)

    glob_check(args.chime7_root, bool(args.has_eval), checksum_json)

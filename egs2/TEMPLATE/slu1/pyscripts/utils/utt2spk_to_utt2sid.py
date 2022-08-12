#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generate utt2sid file from utt2spk and spk2sid files."""

import argparse
import codecs


def main():
    """Print utt2sid in stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("spk2sid", type=str, help="Kaldi-style spk2sid file path.")
    parser.add_argument("utt2spk", type=str, help="Kaldi-style utt2spk file path.")
    args = parser.parse_args()

    # load files
    with codecs.open(args.spk2sid, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    spk2sid = {line.split()[0]: line.split()[1] for line in lines}
    with codecs.open(args.utt2spk, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    utt2spk = {line.split()[0]: line.split()[1] for line in lines}

    for utt_id, spk in utt2spk.items():
        sid = spk2sid.get(spk, 0)
        print(f"{utt_id} {sid}")


if __name__ == "__main__":
    main()

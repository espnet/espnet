#!/usr/bin/env python3

# Copyright 2022 Hitachi LTD (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import os

from espnet2.text.cleaner import TextCleaner

# This is in case of using english_us_mfa as g2p
# Hardcoded part for replaces in LJ. MFA G2P cannot recognize them and generates <spn> labels for the whole word.
# A new G2P model for MFA may need to be trained
# REPLACES = {
#     "'britannia,'" : "britannia,",
#     "'flowery land,'" : "flowery land,",
#     "'flowery land'" : "flowery land",
#     "ll.d" : "l l. d",
#     "i.q" : "i. q",
#     "'smashers,'" : "smashers,",
#     "'lennie'" : "lennie",
# }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--cleaner", type=str, default=None, help="Cleaner type.")
    parser.add_argument("input_file", type=str, default="")
    parser.add_argument("output_dir", type=str, default="")
    args = parser.parse_args()

    cleaner = None
    if args.cleaner is not None:
        cleaner = TextCleaner(args.cleaner)
    with codecs.open(args.input_file, encoding="utf8") as f:
        lines = [line.strip() for line in f.readlines()]
    text = {line.split()[0]: " ".join(line.split()[1:]) for line in lines}
    if cleaner is not None:
        text = {k: cleaner(v) for k, v in text.items()}

    for utt_id, cleaned in text.items():
        outfile = os.path.join(args.output_dir, f"{utt_id}.lab")
        with codecs.open(outfile, "w", encoding="utf8") as writer:
            writer.write(cleaned.lower() + "\n")
    
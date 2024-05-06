import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from prepare_lm_data import cjk2unit, read_text, unit2cjk

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert generated TTS tokens to discrete speech tokens.\
        Only needed if converted to cjk tokens."
    )

    parser.add_argument("-i", "--input", type=Path, required=True, help="Input file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="tts_",
        help="Only keep lines with the specified prefix.",
    )

    args = parser.parse_args()

    fw = open(args.output, "w")

    with open(args.input) as f:
        for line in f:
            line_split = line.split("\t")
            tok = line_split[0].strip()

            utt_id = line_split[1].strip().replace("(", "").replace(")", "")
            if args.prefix:
                utt_id = utt_id.replace("{}".format(args.prefix), "")

            new_tok_list = []
            for t in tok:
                new_tok_list.append(cjk2unit(t))

            fw.write("{} {}\n".format(utt_id, " ".join(new_tok_list)))

    fw.close()

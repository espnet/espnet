#!/usr/bin/env python3
import argparse
import logging

import numpy as np

from espnet2.fileio.read_text import load_num_sequence_text

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("unique_pseudo_labels")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_label", type=str, default=None,
    )
    parser.add_argument(
        "--output_label", type=str, default=None,
    )

    return parser


def main(args):
    input_scp = load_num_sequence_text(args.input_label, loader_type="text_int")
    with open(args.output_label, "w", encoding="utf-8") as fout:
        for key in input_scp.keys():
            value_seq = list(map(str, np.unique(input_scp[key])))
            fout.write("{} {}\n".format(key, " ".join(value_seq)))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.info(str(args))
    
    main(args)

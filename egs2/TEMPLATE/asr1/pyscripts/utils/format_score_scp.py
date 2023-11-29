#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

from espnet.utils.cli_utils import get_commandline_args


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Copy score from "score.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="score",
        help="Specify the prefix word of output file name " 'such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    args = parser.parse_args()

    if args.segments is not None:
        scpfile = Path(args.outdir) / f"{args.name}.scp"
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        fscp = scpfile.open("w", encoding="utf-8")
        dic = {}
        with open(args.scp) as fpath:
            for line in fpath:
                utt_id, path = line.strip().split()
                dic[utt_id] = path
        with open(args.segments) as f:
            for line in f:
                if len(line) == 0:
                    continue
                utt_id, _, _, _ = line.strip().split(" ")
                fscp.write(f"{utt_id} {dic[utt_id]}\n")
        fscp.close()

    else:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        os.system("cp {} {}".format(args.scp, Path(args.outdir) / f"{args.name}.scp"))


if __name__ == "__main__":
    main()

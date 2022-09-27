#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import os

from espnet.utils.cli_utils import get_commandline_args

def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Copy xmls from "xml_dump"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="xml",
        help="Specify the prefix word of output file name " 'such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_midiscp = Path(args.outdir) / f"{args.name}.scp"

    # TODO(Yuning): use sub xmls copied from origin musicxml 
    # More details need to be completed
    # NOTE(Yuning): naXmls are read and written in stage 1: prep_segment_from_xml
    # Option: add XMLScpReader and XMLScpWriter in fileio/xml_scp.py
    if args.segments is not None:
        scpfile = Path(args.outdir) / f"{args.name}.scp"
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        fscp = scpfile.open("w", encoding="utf-8")
        with open(args.segments) as f:
            for line in f:
                if len(line) == 0:
                    continue
                utt_id, _, _, _ = line.strip().split(" ")
                xml_path = Path(args.outdir) / f"{utt_id}.{args.name}"
                xml_path.parent.mkdir(parents=True, exist_ok=True)
                os.system("cp {}/{}.musicxml {}".format(args.scp, utt_id, xml_path))
                fscp.write(f"{utt_id} {xml_path}\n")
        fscp.close()

    else:
        # TODO(Yuning): xml_scp without segment
        raise IOError("No segment file.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import io
import sys

PY2 = sys.version_info[0] == 2

if PY2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Mixing wav.scp files into a multi-channel wav.scp " "using sox.",
    )
    parser.add_argument("scp", type=str, nargs="+", help="Give wav.scp")
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    fscps = [io.open(scp, "r", encoding="utf-8") for scp in args.scp]
    for linenum, lines in enumerate(zip_longest(*fscps)):
        keys = []
        wavs = []

        for line, scp in zip(lines, args.scp):
            if line is None:
                raise RuntimeError("Numbers of line mismatch")

            sps = line.split(" ", 1)
            if len(sps) != 2:
                raise RuntimeError(
                    'Invalid line is found: {}, line {}: "{}" '.format(
                        scp, linenum, line
                    )
                )
            key, wav = sps
            keys.append(key)
            wavs.append(wav.strip())

        if not all(k == keys[0] for k in keys):
            raise RuntimeError(
                "The ids mismatch. Hint; the input files must be "
                "sorted and must have same ids: {}".format(keys)
            )

        args.out.write(
            "{} sox -M {} -c {} -t wav - |\n".format(
                keys[0], " ".join("{}".format(w) for w in wavs), len(fscps)
            )
        )


if __name__ == "__main__":
    main()

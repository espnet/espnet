from __future__ import print_function
from itertools import izip
import argparse
import sys
import os
import codecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Src aug file", type=str)
    parser.add_argument("tgt", help="tgt aug file", type=str)
    parser.add_argument("--max-len", help="max length", type=int,
                        default=250, action="store")
    args = parser.parse_args()

    fo1 = codecs.open(args.src + ".filt", "w", encoding="utf-8")
    fo2 = codecs.open(args.tgt + ".filt", "w", encoding="utf-8")

    with codecs.open(args.src, "r", encoding="utf-8") as fsi:
        with codecs.open(args.tgt, "r", encoding="utf-8") as fti:
            l = 0
            for l1, l2 in izip(fsi, fti):
                print("\rLine, ", l, end="") 
                if (len(l2.strip().split()) <= args.max_len):
                    print(l1.strip(), file=fo1)
                    print(l2.strip(), file=fo2)
                l += 1


if __name__ == "__main__":
    main()




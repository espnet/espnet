#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    texts = {}

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            entry = line.split(" ")
            eid = entry[0]
            trans = " ".join(entry[1:]).rstrip()
            tid = eid.split("_")[0]
            if tid in texts:
                texts[tid] += trans
            else:
                texts[tid] = trans
            line = f.readline()

    with open(sys.argv[2], "w", encoding="utf-8") as fw:
        for k, v in texts.items():
            print("{} {}".format(k, v), file=fw)

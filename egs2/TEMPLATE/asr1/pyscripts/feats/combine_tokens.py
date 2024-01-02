#!/usr/bin/env python

# Copyright 2023 Yuning Wu

import argparse

from espnet2.fileio.read_text import load_num_sequence_text


def combine(args):
    dic = {}
    tokens = args.tokens.split(' ')
    for token in tokens:
        path = args.dir+ 'token_' + token
        dic[token] = load_num_sequence_text(path, "text_int")
    f = open(args.dir + args.target, 'w')
    for idx in dic[tokens[0]].keys():
        l = min([len(dic[token][idx]) for token in tokens])
        lis = []
        if args.mix_type == "sequence":
            for token in tokens:
                lis.extend(dic[token][idx][: l])
        elif args.mix_type == "frame":
            for i in range(l):
                for token in tokens:
                    lis.append(dic[token][idx][i])
        f.write("{} {}\n".format(idx, " ".join(str(v) for v in lis)))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multi discrete tokens.")
    parser.add_argument("--dir", type=str, help="data directory")
    parser.add_argument("--tokens", type=str, default="discrete tokens listed as type/layer")
    parser.add_argument("--target", type=str, help="target token file")
    parser.add_argument("--mix_type", type=str, default="frame", help="target token file")
    args = parser.parse_args()
    combine(args)
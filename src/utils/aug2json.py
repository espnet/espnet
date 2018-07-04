#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--aug", action="store", required=True,
                        help="Augmenting data")
    parser.add_argument("-d", "--dict", action="store", required=True,
                        help="Augmenting data dictionary prefix")
    parser.add_argument("-j", "--json", action="store", required=True,
                        help="Other json with which to merge (overwrite)")

    args = parser.parse_args()

    aug_json = {}
    odict_lines = [(l.strip().split()) for l in open(args.dict, 'r').readlines()]
    idict = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
    odict = {k: int(v) for k, v in odict_lines}
    ofilename = args.aug + ".tgt"
    ifilename = args.aug + ".src"
    aug_json['aug'] = {}
    aug_json['aug']['ifilename'] = ifilename
    aug_json['aug']['ofilename'] = ofilename
    aug_json['aug']['odict'] = odict
    aug_json['aug']['sentences'] = {}

    with open(ofilename, "r") as fo:
        with open(ifilename, "r") as fi:
            ooffset = fo.tell()
            ioffset = fi.tell()
            line_num = 0
            iline = fi.readline()
            oline = fo.readline()
            while(iline and oline):
                iline_toks = iline.strip().split()
                for i_tok in iline_toks:
                    idict[i_tok] = idict.get(i_tok, len(idict))
                olen = len(oline.strip().split())
                ilen = len(iline_toks)
                aug_json['aug']['sentences'][line_num] = {
                    'ilen': ilen,
                    'olen': olen,
                    'ioffset': ioffset,
                    'ooffset': ooffset
                }
                ooffset = fo.tell()
                ioffset = fi.tell()
                iline = fi.readline()
                oline = fo.readline()
                line_num += 1
    print('using ' + str(line_num) + ' lines of augmenting data', file=sys.stderr)
    aug_json['aug']['idict'] = idict  # input side dictionary with <pad> symbol
    with open(args.json, "w") as f:
        json.dump(aug_json, f, indent=4)
    for k, v in idict.items():
        print(k + ' ' + str(v))


if __name__ == "__main__":
    main()

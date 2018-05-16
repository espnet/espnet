#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
import codecs
import json
import argparse
import numpy as np
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="data.json file with olens and ilens")
    parser.add_argument("phone_txt", help="phonetic transcription of text data")
    parser.add_argument("output", help="output")

    args = parser.parse_args()

    with codecs.open(args.json, "r", encoding="utf-8") as f:
        j = json.load(f)

    stretch_factors = []
    for key, u in j['utts'].iteritems():
        stretch_factors.append(float(u['ilen']) / float(u['olen']))
    mu = np.mean(stretch_factors)
    sig = np.var(stretch_factors)
    
    count = 1
    with codecs.open(args.output, "w", encoding="utf-8") as fo: 
        with codecs.open(args.phone_txt, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip() != '':
                    print("\rCount: ", count, end="")
                    phones = l.strip().split()
                    phn_mu = mu
                    phn_sig = sig / np.sqrt(float(len(phones)))
                    for p in phones:
                        for i in range(max(1,
                                           int(round(np.random.normal(loc=phn_mu,
                                                                      scale=abs(np.random.normal(loc=phn_sig, scale=2.0)))/4.0)))):
                            print(p, end=" ", file=fo)
                    print(file=fo)
                count += 1

print()

if __name__ == "__main__":
    main()

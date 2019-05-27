#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text')
    parser.add_argument('file_order', type=str,
                        help='text file which describes the order of audio files')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    file_order = []
    with open(args.file_order, 'r') as f:
        for line in f:
            file_order.append(line.strip().replace('.en', ''))

    logging.info("reading %s", args.text)
    with open(args.text, 'r') as f:
        refs = f.readlines()

    dic = {}
    for line in refs:
        utt_id = line.split()[0]
        talk_id = utt_id.split('_')[0].replace('.en', '').replace('.de', '')
        ref = ' '.join(line.split()[1:])

        if talk_id not in dic.keys():
            dic[talk_id] = []
        dic[talk_id] += [ref]

    for talk_id in file_order:
        for ref in dic[talk_id]:
            # print(talk_id + ' ' + ref)
            print(ref)

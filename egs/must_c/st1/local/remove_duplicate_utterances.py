#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
from collections import OrderedDict
import logging
# import editdistance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.text)
    wav_dict = OrderedDict()
    start_prev = 0
    end_prev = 0
    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            utt_id = line.split()[0]
            utt_id_no_lang_id = '-'.join(utt_id.split('-')[:-1])
            session = utt_id_no_lang_id.split('_')[0]
            start = int(utt_id_no_lang_id.split('_')[-2])
            end = int(utt_id_no_lang_id.split('_')[-1])
            ref = ' '.join(line.split()[1:])

            # We regard as the same segment when the difference of start time is < 10
            if session not in wav_dict.keys():
                wav_dict[session] = OrderedDict()
                wav_dict[session][start] = (end, utt_id, ref)
            elif start not in wav_dict[session].keys():
                if abs(start - start_prev) < 10:
                    if end > end_prev:
                        # keep the longer utterance
                        wav_dict[session][start] = (end, utt_id, ref)
                    else:
                        continue
                else:
                    wav_dict[session][start] = (end, utt_id, ref)

            start_prev = start
            end_prev = end

    for session in wav_dict.keys():
        for (end, utt_id, ref) in wav_dict[session].values():
            print(utt_id + ' ' + ref)

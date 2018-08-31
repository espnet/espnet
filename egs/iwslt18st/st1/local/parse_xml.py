#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import re
from xml.etree.ElementTree import parse

from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    args = parser.parse_args()

    tree = parse(args.text)
    elem = tree.getroot()

    set = os.path.basename(args.text).split('.')[2]
    lang = os.path.basename(args.text).split('.')[-2]
    talk_id = None

    # Parse a XML file
    trans_dict_all = OrderedDict()
    for e in elem.getiterator():
        if e.tag == 'doc':
            talk_id = e.get('docid').replace(' ', '')
            trans_dict_all[talk_id] = OrderedDict()
        elif e.tag == 'seg':
            utt_id = int(e.get('id'))
            trans = e.text.encode('utf-8')

            # Remove consecutive spaces
            trans = re.sub(r'[\s]+', ' ', trans)

            # Remove the first space
            if trans[0] == ' ':
                trans = trans[1:]

            # Remove the last space
            if trans[-1] == ' ':
                trans = trans[:-1]

            trans_dict_all[talk_id][utt_id] = trans

    for talk_id, trans_dict in trans_dict_all.items():
        for utt_id, trans in trans_dict.items():
            print("%s.%s.talkid%d_%04d %s" % (set, lang, int(talk_id), int(utt_id), trans))


if __name__ == '__main__':
    main()

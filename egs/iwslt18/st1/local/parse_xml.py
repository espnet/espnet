#!/usr/bin/env python
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
from collections import OrderedDict
import os
import re
import xml.etree.ElementTree as etree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml', type=str, default=False, nargs='?',
                        help='input xml')
    parser.add_argument('output', type=str, default=False, nargs='?',
                        help='output text')
    args = parser.parse_args()

    with codecs.open(args.xml, 'r', encoding="utf-8") as xml_file:
        elem = etree.parse(xml_file).getroot()

        _set = os.path.basename(args.xml).split('.')[2]
        lang = os.path.basename(args.xml).split('.')[-2]
        talk_id = None

        # Parse a XML file
        trans_dict_all = OrderedDict()
        for e in elem.getiterator():
            if e.tag == 'doc':
                talk_id = e.get('docid').replace(' ', '')
                trans_dict_all[talk_id] = OrderedDict()
            elif e.tag == 'seg':
                utt_id = int(e.get('id'))
                ref = e.text

                # Remove Al Gore:, Video: etc.
                # ref = ref.split(':')[-1].lstrip()

                # Remove consecutive spaces
                ref = re.sub(r'[\s]+', ' ', ref).lstrip().rstrip()

                trans_dict_all[talk_id][utt_id] = ref

    with codecs.open(args.output, 'w', encoding="utf-8") as f:
        for talk_id, trans_dict in trans_dict_all.items():
            for utt_id, ref in trans_dict.items():
                f.write("%s.%s.talkid%d_%04d %s\n" % (_set, lang, int(talk_id), int(utt_id), ref))


if __name__ == '__main__':
    main()

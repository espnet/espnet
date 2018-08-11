#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import unicode_literals

import argparse
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    args = parser.parse_args()

    if args.text:
        f = open(args.text, "rb")
    else:
        f = sys.stdin

    i = 0
    for line in f:
        x = unicode(line, 'utf_8').strip()

        # Remove disallowed Unicode whitespaces :\x{0009}\x{000a}\x{0020}
        x = x.replace(' ', ' ')

        # TODO(hirofumi): Remove punctuation marks here

        # Remove other unnecessary marks
        # x = x.replace('–', '')
        # x = x.replace('—', '')  # different from above
        # x = x.replace('…', '')
        # x = x.replace('♪', '')
        # x = x.replace('♫', '')
        # [][!?;-\"

        # Remove consecutive spaces
        x = re.sub(r'[\s]+', ' ', x)

        print x.encode('utf-8')

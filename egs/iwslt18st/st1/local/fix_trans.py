#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import unicode_literals

import argparse
import re
import sys

WHITESPACE = ' '

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    args = parser.parse_args()

    if args.text:
        f = open(args.text, "rb")
    else:
        f = sys.stdin

    for line in f:
        x = unicode(line, 'utf_8').strip()

        # ted_2128_0054026_0054442 auf 40 kilowatt[stunden]  pro quadratmeter im jahr.
        x = x.replace(' ', WHITESPACE)

        # TODO(hirofumi): ted_1890_0010122_0011316 und obwohl es stimmt, dass afrika ein rauer ort ist, kenne ich es auch als einen ort, an dem menschen, tiere undÖkosysteme uns von einer verbundenen welt erzählen.

        # TODO(hirofumi): Remove punctuation marks here

        # TODO(hirofumi): Remove other unnecessary marks here

        # Remove consecutive whitespaces
        x = re.sub(r'[\s]+', WHITESPACE, x)

        print x.encode('utf-8')

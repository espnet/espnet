#!/usr/bin/env python3
# encoding: utf-8

import argparse
from io import open
import json
import logging
import sys
import codecs

# I wonder if I can successfully import this line...
from espnet.utils.cli_utils import get_commandline_args

from text_cleaner import remove
from text_cleaner.processor.common import ASCII, SYMBOLS_AND_PUNCTUATION_EXTENSION, GENERAL_PUNCTUATION
from text_cleaner.processor.chinese import CHINESE, CHINESE_SYMBOLS_AND_PUNCTUATION

def get_parser():
    parser = argparse.ArgumentParser(
        description='Clean transcription file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('transcription', type=str,
                        help='Transcription file')
    parser.add_argument('utt2spk', type=str, help='utt2spk file for the speaker')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())
    
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    # read transcription file
    transcriptions = {}
    with codecs.open(args.transcription, 'r', 'utf-8') as fid:
        for l in fid.read().splitlines():
            line = l.split(" ")
            lang_char = args.transcription.split('/')[-1][0]
            id = lang_char + line[0] # ex. E10001
            text = line[1]
            text = CHINESE_SYMBOLS_AND_PUNCTUATION.remove(text)
            text = SYMBOLS_AND_PUNCTUATION_EXTENSION.remove(text)
            text = GENERAL_PUNCTUATION.remove(text)
            text = text.replace(" ", "")
            transcriptions[id] = text
    print(transcriptions)
    
    # read the utt2spk file and actually write
    with codecs.open(args.utt2spk, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            segments = line.split(" ")
            id = segments[0] # ex. E10001
            content = transcriptions[id]

            print("%s %s" % (id, content))

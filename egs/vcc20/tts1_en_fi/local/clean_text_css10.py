#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import os

from text.cleaners import (lowercase, expand_numbers, expand_abbreviations,
                           expand_symbols, remove_unnecessary_symbols,
                           uppercase, collapse_whitespace)

try:
    # For phoneme conversion, use https://github.com/Kyubyong/g2p.
    from g2p_en import G2p
    f_g2p = G2p()
    f_g2p("")
except ImportError:
    raise ImportError("g2p_en is not installed. please run `. ./path.sh && pip install g2p_en`.")
except LookupError:
    # NOTE: we need to download dict in initial running
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt")


def custom_finnish_cleaners(text):
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = expand_symbols(text)
    text = remove_unnecessary_symbols(text)
    text = uppercase(text)
    text = collapse_whitespace(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_tag", type=str, default=None, nargs="?",
                        help="language tag (can be used for multi lingual case)")
    parser.add_argument("--spk_tag", type=str,
                        help="speaker tag")
    parser.add_argument("transcription", type=str,
                        help="transcription filename")
    parser.add_argument("out", type=str,
                        help="output filename")
    parser.add_argument("trans_type", type=str, default="char",
                        help="transcription type (char or phn)")
    args = parser.parse_args()

    dirname = os.path.dirname(args.out)
    if len(dirname) != 0 and not os.path.exists(dirname):
        os.makedirs(dirname)

    with codecs.open(args.out, "w", encoding="utf-8") as out:
        with codecs.open(args.transcription, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                path, _, content, _ = line.split('|')
                uid = args.spk_tag + "_" + os.path.basename(path).replace(".wav", "")
                text = custom_finnish_cleaners(content.rstrip())
                if args.lang_tag is None:
                    line = "%s %s\n" % (uid, text)
                else:
                    line = "%s <%s> %s\n" % (uid, args.lang_tag, text)
                out.write(line)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import os

from text.cleaners import custom_english_cleaners
import nltk

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


def g2p(text):
    """Convert grapheme to phoneme."""
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return ' '.join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_tag", type=str, default=None, nargs="?",
                        help="language tag (can be used for multi lingual case)")
    parser.add_argument("--spk_tag", type=str,
                        help="speaker tag")
    parser.add_argument("jsons", nargs="+", type=str,
                        help="*_mls.json filenames")
    parser.add_argument("out", type=str,
                        help="output filename")
    parser.add_argument("trans_type", type=str, default="phn",
                        choices=["char", "phn"],
                        help="Input transcription type")
    args = parser.parse_args()

    dirname = os.path.dirname(args.out)
    if len(dirname) != 0 and not os.path.exists(dirname):
        os.makedirs(dirname)

    with codecs.open(args.out, "w", encoding="utf-8") as out:
        for filename in sorted(args.jsons):
            with codecs.open(filename, "r", encoding="utf-8") as f:
                js = json.load(f)
            for key in sorted(js.keys()):
                uid = args.spk_tag + "_" + key.replace(".wav", "")
                
                content = js[key]["clean"]
                text = custom_english_cleaners(content.rstrip())
                if args.trans_type == "phn":
                    clean_content = text.lower()
                    text = g2p(clean_content)

                if args.lang_tag is None:
                    line = "%s %s \n" % (uid, text)
                else:
                    line = "%s <%s> %s\n" % (uid, args.lang_tag, text)
                out.write(line)


if __name__ == "__main__":
    main()

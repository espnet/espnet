#!/usr/bin/env python3

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import nltk
from tacotron_cleaner.cleaners import custom_english_cleaners

try:
    # For phoneme conversion, use https://github.com/Kyubyong/g2p.
    from g2p_en import G2p

    f_g2p = G2p()
    f_g2p("")
except ImportError:
    raise ImportError(
        "g2p_en is not installed. please run `. ./path.sh && pip install g2p_en`."
    )
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
    return " ".join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "transcription_path", type=str, help="path for the transcription text file"
    )
    parser.add_argument(
        "--lang_tag",
        type=str,
        default="",
        help="language tag (can be used for multi lingual case)",
    )
    parser.add_argument(
        "--trans_type",
        type=str,
        default="char",
        choices=["char", "phn"],
        help="Input transcription type",
    )
    parser.add_argument(
        "--lowercase", type=bool, default=False, help="Lower case the result or not"
    )
    args = parser.parse_args()

    # clean every line in transcription file first
    with codecs.open(args.transcription_path, "r", "utf-8") as fid:
        for line in fid.read().splitlines():
            segments = line.split(" ")

            # clean contents
            content = " ".join(segments[:-1])
            clean_content = custom_english_cleaners(content)

            # get id by taking off the parentheses
            id = segments[-1][1:-1]

            if args.trans_type == "phn":
                text = clean_content.lower()
                clean_content = g2p(text)

            if args.lowercase:
                clean_content = clean_content.lower()

            if args.lang_tag == "":
                print("{} {}".format(id, clean_content))
            else:
                print("{} {}".format(id, "<" + args.lang_tag + "> " + clean_content))

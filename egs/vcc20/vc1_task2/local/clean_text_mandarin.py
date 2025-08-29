#!/usr/bin/env python3

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import nltk
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
from pypinyin.style._utils import get_finals, get_initials
from tacotron_cleaner.cleaners import custom_english_cleaners


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


my_pinyin = Pinyin(MyConverter())
pinyin = my_pinyin.pinyin

E_lang_tag = "en_US"

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
    parser.add_argument("utt2spk", type=str, help="utt2spk file for the speaker")
    parser.add_argument(
        "trans_type",
        type=str,
        default="phn",
        choices=["char", "phn"],
        help="Input transcription type",
    )
    parser.add_argument("lang_tag", type=str, help="lang tag")
    parser.add_argument("spk", type=str, help="speaker name")
    parser.add_argument(
        "--transcription_path_en",
        type=str,
        default=None,
        help="path for the English transcription text file",
    )
    args = parser.parse_args()

    # clean every line in transcription file first
    transcription_dict = {}
    with codecs.open(args.transcription_path, "r", "utf-8") as fid:
        for line in fid.readlines():
            segments = line.split(" ")
            lang_char = args.transcription_path.split("/")[-1][0]
            id = args.spk + "_" + lang_char + segments[0]  # ex. TMF1_M10001
            content = segments[1].replace("\n", "")

            # Some special rules to match CSMSC pinyin
            text = pinyin(content, style=Style.TONE3)
            text = [c[0] for c in text]
            clean_content = []
            for c in text:
                c_init = get_initials(c, strict=True)
                c_final = get_finals(c, strict=True)
                for c in [c_init, c_final]:
                    if len(c) == 0:
                        continue
                    c = c.replace("ü", "v")
                    c = c.replace("ui", "uei")
                    c = c.replace("un", "uen")
                    c = c.replace("iu", "iou")

                    # Special rule: "e5n" -> "en5"
                    if "5" in c:
                        c = c.replace("5", "") + "5"
                    clean_content.append(c)

            transcription_dict[id] = " ".join(
                ["<" + args.lang_tag + ">"] + clean_content
            )

    if args.transcription_path_en:
        with codecs.open(args.transcription_path_en, "r", "utf-8") as fid:
            for line in fid.readlines():
                segments = line.split(" ")
                id = args.spk + "_" + "E" + segments[0]  # ex. TMF1_E10001
                content = " ".join(segments[1:])
                clean_content = custom_english_cleaners(content.rstrip())
                if args.trans_type == "phn":
                    text = clean_content.lower()
                    clean_content = g2p(text)

                transcription_dict[id] = "<" + E_lang_tag + "> " + clean_content

    # read the utt2spk file and actually write
    with codecs.open(args.utt2spk, "r", "utf-8") as fid:
        for line in fid.readlines():
            segments = line.split(" ")
            id = segments[0]  # ex. E10001
            content = transcription_dict[id]

            print("%s %s" % (id, content))

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import re

import regex

parser = argparse.ArgumentParser()
parser.add_argument("text", type=str, help="text file")
args = parser.parse_args()


def main():
    p_kanji = regex.compile(r".*\p{Script=Han}+.*")
    p_hiragana = regex.compile(r".*\p{Block=Hiragana}+.*")
    p_katakana = regex.compile(r".*\p{Block=Katakana}+.*")
    p_chinese = re.compile(".*[\u4e00-\u9fa5]+.*")
    p_korean = re.compile(".*[\uac00-\ud7ff]+.*")
    p_arabic = regex.compile(r".*\p{Block=Arabic}+.*")
    p_cyrillic = regex.compile(r".*\p{Block=Cyrillic}+.*")
    p_sanskrit = regex.compile(r".*\p{Block=Devanagari}+.*")
    p_egyptian = regex.compile(r".*\p{Block=Egyptian_Hieroglyphs}+.*")
    p_ethiopic = regex.compile(r".*\p{Block=Ethiopic}+.*")
    p_hebrew = regex.compile(r".*\p{Block=Hebrew}+.*")
    p_armenian = regex.compile(r".*\p{Block=Armenian}+.*")
    p_thai = regex.compile(r".*\p{Block=Thai}+.*")
    p_bengali = regex.compile(r".*\p{Block=Bengali}+.*")
    p_myanmer = regex.compile(r".*\p{Block=Myanmar}+.*")
    p_geogian = regex.compile(r".*\p{Block=Georgian}+.*")
    p_lao = regex.compile(r".*\p{Block=Lao}+.*")

    # exception
    def is_dhivehi(text):
        return "މާވަށް" in text

    with codecs.open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            sentence = " ".join(line.split(" ")[1:])
            if (
                p_kanji.match(sentence) is None
                and p_hiragana.match(sentence) is None
                and p_katakana.match(sentence) is None
                and p_chinese.match(sentence) is None
                and p_korean.match(sentence) is None
                and p_arabic.match(sentence) is None
                and p_cyrillic.match(sentence) is None
                and p_sanskrit.match(sentence) is None
                and p_egyptian.match(sentence) is None
                and p_ethiopic.match(sentence) is None
                and p_hebrew.match(sentence) is None
                and p_armenian.match(sentence) is None
                and p_thai.match(sentence) is None
                and p_bengali.match(sentence) is None
                and p_myanmer.match(sentence) is None
                and p_geogian.match(sentence) is None
                and p_lao.match(sentence) is None
                and not is_dhivehi(sentence)
            ):
                print(line)


if __name__ == "__main__":
    main()

# coding: utf-8

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import re
import sys

import MeCab
import mojimoji

from pykakasi import kakasi


class Text2Yomi(object):
    """Converter from text to yomigana in hiragana

    :param str dict_dir: dictionay directory
    """

    def __init__(self, dict_dir):
        self.tagger = MeCab.Tagger('-d ' + dict_dir)
        self.kakasi = kakasi()
        self.kakasi.setMode('K', 'H')
        self.conv = self.kakasi.getConverter()

    def __call__(self, text):
        """Convert text to yomigana

        :param str text: input text including kanji, katakana and hiragana
        :return yomigana in hiragana
        :rtype str
        """
        text = mojimoji.zen_to_han(text, kana=False)
        if sys.version_info.major == 2:
            text = text.encode('utf-8')
        text = self.tagger.parse(text).split("\n")[:-2]
        new_text = []
        prev_word = ''
        for word in text:
            word = word.replace("\n", '').split(',')
            if word[-1] == '*':
                tmp = word[0].split("\t")[0]
                if self.is_hiragana(tmp):
                    word = tmp
                elif tmp == 'ー':
                    word = prev_word + tmp
                    new_text[-1] = word
                    prev_word = word
                    continue
                else:
                    word = '*'
            else:
                word = word[-1]
            new_text += [word]
            prev_word = word

        joined_text = ' '.join(new_text)
        if sys.version_info.major == 2:
            joined_text = joined_text.decode('utf-8')

        return self.conv.do(joined_text)

    def is_hiragana(self, text):
        return re.search('[あ-ん]', text)

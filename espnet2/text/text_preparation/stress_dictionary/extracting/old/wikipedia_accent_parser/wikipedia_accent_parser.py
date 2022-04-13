#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
from collections import Counter


ACCENT = u'́'


class WikipediaAccentParser:

    def get_dump_path(self):
        filename = 'wiki_mini_dump.json'

        folder = os.path.dirname(sys.modules['wikipedia_accent_parser'].__file__)

        path = os.path.join(folder, filename)

        return path

    def __init__(self):

        self.accented_words_by_not_accented = {}

        with open(self.get_dump_path()) as file:
            wiki_dump = json.load(file)

        for article in wiki_dump:
            title = article['title']
            accented_items = article['accented_items']

            for accented_item in accented_items:
                tokens = accented_item.split()
                for token in tokens:
                    normalized_accented_token = self.normalize_word(token)
                    normalized_token = normalized_accented_token.replace(ACCENT, "")

                    if normalized_token not in self.accented_words_by_not_accented:
                        self.accented_words_by_not_accented[normalized_token] = []

                    self.accented_words_by_not_accented[normalized_token].append({
                        "accented_option": normalized_accented_token,
                        "source_title": title
                    })

    def normalize_word(self, accented_item):
        return accented_item.lower().replace(u"ё", u"е")

    def retrieve_accent(self, word, detailed_info=False):
        normalized_word = self.normalize_word(word)
        options = self.accented_words_by_not_accented.get(normalized_word)

        if options is None:
            return normalized_word, [] if detailed_info else normalized_word

        count_by_option = Counter()

        for option in options:
            count_by_option[option["accented_option"]] += 1

        answer = count_by_option.most_common()[0][0]

        if detailed_info:
            return answer, options
        else:
            return answer

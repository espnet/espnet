#!/usr/bin/env python3.7

"""
Parsing wikipedia dump file.

It can be downloaded here: https://dumps.wikimedia.org/wikidatawiki/entities/
File name is `latest-lexemes.nt.bz2`
"""

import codecs
import operator
import re
import stresses_lib as lib
import sys

assert len(sys.argv) == 2, "Wrong number of arguments"
lexems_fn = sys.argv[1]


STRESSED_WORD_PATTERN = re.compile(r'[а-яА-Я]*[ёЁ\u0301][а-яА-ЯёЁ]*')


def extract_accents(raw_str):
    s = raw_str.lower()
    # Some words like 'сёгун' have other then 'ё' stressed,
    # so find explicit stress first
    accent_idx = s.find('\u0301')
    if accent_idx == -1:
        accent_idx = s.index('ё') + 1
        clean_str = s
    else:
        clean_str = s.replace('\u0301', '')
    # Wikipedia has some malformed string, i.e.
    # 'Плотников´Иван Васильевич' or
    # 'по поводу конкретного Ив́ан Ив́ановича Ив́анова'
    acceptable = len(clean_str) > 1 \
        and accent_idx > 0 \
        and lib.is_vowel(clean_str[accent_idx - 1])
    return acceptable, accent_idx, clean_str


lc = 0
words = {}
with open(lexems_fn, 'r') as f:
    for line in f:
        lc += 1
        # if lc > 1000000:
        #     break
        for acc_str in STRESSED_WORD_PATTERN.findall(line):
            acceptable, i, s = extract_accents(acc_str)
            if acceptable:
                # if s == "иван":
                #     print("--------------")
                #     print(i)
                #     print(line)
                poses = words.get(s, {})
                cnt = poses.get(i, 0)
                poses[i] = cnt + 1
                words[s] = poses

# Now calculate most frequent stresses
words = {word: [max(stresses.items(), key=operator.itemgetter(1))[0]]
         for word, stresses in words.items()}
# And finally save result
lib.save_stresses(words)

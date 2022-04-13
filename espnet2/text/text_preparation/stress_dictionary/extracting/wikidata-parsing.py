#!/usr/bin/env python3.7
#
# Parsing wikidata lexems file.
# It can be downloaded here: https://dumps.wikimedia.org/wikidatawiki/entities/
# File name is `latest-lexemes.nt.bz2`
#

import codecs
import re
import sys
import stresses_lib as lib

assert len(sys.argv) == 2, "Wrong number of arguments"
lexems_fn = sys.argv[1]

RU_WORDS_PATTERN = re.compile(r'"(.*\\u(0301|0401|0451).*)"@ru')
xp = re.compile(r'^[а-яё-]+$')


def extract_accents(raw_str):
    decoded_str = codecs.decode(raw_str, 'unicode_escape')
    if ' ' in decoded_str:
        # Some 'lexems' in Wikitionary contains whole phrases; skip it
        return False, 0, 0, 0
    else:
        preaccented_str = decoded_str \
            .replace("Ю́", "ю\u0301") \
            .replace("О́", "о\u0301") \
            .replace("ё", "ё\u0301")
        preclear_str = preaccented_str \
            .replace('\u0300', '') \
            .translate(str.maketrans("ѐѝЗ", "еиз"))
        accent_idx = preclear_str.index('\u0301')
        clear_str = preclear_str.replace('\u0301', '')
        return True, accent_idx, decoded_str, clear_str


words = {}
with open(lexems_fn, 'r') as f:
    for line in f:
        m = RU_WORDS_PATTERN.search(line)
        if m:
            acceptable, i, d, s = extract_accents(m.group(1))
            if acceptable:
                if not xp.match(s):
                    pass
                poses = words.get(s, set())
                poses.add(i)
                words[s] = poses

lib.save_stresses(words)

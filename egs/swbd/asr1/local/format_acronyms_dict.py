#!/usr/bin/env python3

# Copyright 2015  Minhua Wu
# Apache 2.0

# convert acronyms in swbd dict to fisher convention
# IBM to i._b._m.
# BBC to b._b._c.
# BBCs to b._b._c.s
# BBC's to b._b._c.'s

import argparse
import re

__author__ = "Minhua Wu"

parser = argparse.ArgumentParser(description="format acronyms to a._b._c.")
parser.add_argument("-i", "--input", help="Input lexicon", required=True)
parser.add_argument("-o", "--output", help="Output lexicon", required=True)
parser.add_argument(
    "-L", "--Letter", help="Input single letter pronunciation", required=True
)
parser.add_argument("-M", "--Map", help="Output acronyms mapping", required=True)
args = parser.parse_args()


fin_lex = open(args.input, "r")
fin_Letter = open(args.Letter, "r")
fout_lex = open(args.output, "w")
fout_map = open(args.Map, "w")

# Initialise single letter dictionary
dict_letter = {}
for single_letter_lex in fin_Letter:
    items = single_letter_lex.split()
    dict_letter[items[0]] = single_letter_lex[len(items[0]) + 1 :].strip()
fin_Letter.close()
# print dict_letter

for lex in fin_lex:
    items = lex.split()
    word = items[0]
    lexicon = lex[len(items[0]) + 1 :].strip()
    # find acronyms from words with only letters and '
    pre_match = re.match(r"^[A-Za-z]+$|^[A-Za-z]+\'s$|^[A-Za-z]+s$", word)
    if pre_match:
        # find if words in the form of xxx's is acronym
        if word[-2:] == "'s" and (lexicon[-1] == "s" or lexicon[-1] == "z"):
            actual_word = word[:-2]
            actual_lexicon = lexicon[:-2]
            acronym_lexicon = ""
            for l in actual_word:
                acronym_lexicon = acronym_lexicon + dict_letter[l.upper()] + " "
            if acronym_lexicon.strip() == actual_lexicon:
                acronym_mapped = ""
                acronym_mapped_back = ""
                for l in actual_word[:-1]:
                    acronym_mapped = acronym_mapped + l.lower() + "._"
                    acronym_mapped_back = acronym_mapped_back + l.lower() + " "
                acronym_mapped = acronym_mapped + actual_word[-1].lower() + ".'s"
                acronym_mapped_back = (
                    acronym_mapped_back + actual_word[-1].lower() + "'s"
                )
                fout_map.write(
                    word + "\t" + acronym_mapped + "\t" + acronym_mapped_back + "\n"
                )
                fout_lex.write(acronym_mapped + " " + lexicon + "\n")
            else:
                fout_lex.write(lex)

        # find if words in the form of xxxs is acronym
        elif word[-1] == "s" and (lexicon[-1] == "s" or lexicon[-1] == "z"):
            actual_word = word[:-1]
            actual_lexicon = lexicon[:-2]
            acronym_lexicon = ""
            for l in actual_word:
                acronym_lexicon = acronym_lexicon + dict_letter[l.upper()] + " "
            if acronym_lexicon.strip() == actual_lexicon:
                acronym_mapped = ""
                acronym_mapped_back = ""
                for l in actual_word[:-1]:
                    acronym_mapped = acronym_mapped + l.lower() + "._"
                    acronym_mapped_back = acronym_mapped_back + l.lower() + " "
                acronym_mapped = acronym_mapped + actual_word[-1].lower() + ".s"
                acronym_mapped_back = (
                    acronym_mapped_back + actual_word[-1].lower() + "'s"
                )
                fout_map.write(
                    word + "\t" + acronym_mapped + "\t" + acronym_mapped_back + "\n"
                )
                fout_lex.write(acronym_mapped + " " + lexicon + "\n")
            else:
                fout_lex.write(lex)

        # find if words in the form of xxx (not ended with 's or s) is acronym
        elif word.find("'") == -1 and word[-1] != "s":
            acronym_lexicon = ""
            for l in word:
                acronym_lexicon = acronym_lexicon + dict_letter[l.upper()] + " "
            if acronym_lexicon.strip() == lexicon:
                acronym_mapped = ""
                acronym_mapped_back = ""
                for l in word[:-1]:
                    acronym_mapped = acronym_mapped + l.lower() + "._"
                    acronym_mapped_back = acronym_mapped_back + l.lower() + " "
                acronym_mapped = acronym_mapped + word[-1].lower() + "."
                acronym_mapped_back = acronym_mapped_back + word[-1].lower()
                fout_map.write(
                    word + "\t" + acronym_mapped + "\t" + acronym_mapped_back + "\n"
                )
                fout_lex.write(acronym_mapped + " " + lexicon + "\n")
            else:
                fout_lex.write(lex)
        else:
            fout_lex.write(lex)

    else:
        fout_lex.write(lex)

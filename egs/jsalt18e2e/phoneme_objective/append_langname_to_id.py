"""Converts the IDs in a Babel Kaldi scp-style file so that they are suffixed by the language name."""

import sys

phn_ali_fn = sys.argv[1]

langs = ["101-cantonese", "102-assamese", "103-bengali", "104-pashto", "105-turkish", "106-tagalog",
         "107-vietnamese", "201-haitian", "202-swahili", "203-lao", "204-tamil", "205-kurmanji",
         "206-zulu", "207-tokpisin", "404-georgian"]
langid2name = {}
for lang in langs:
    langid, langname = lang.split("-")
    langid2name[langid] = langname

with open(phn_ali_fn) as f:
    for line in f:
        utterid, *tail = line.split(" ")
        langid = utterid.split("_")[0]
        new_utterid = "{}-{}".format(utterid, langid2name[langid])
        print(new_utterid, *tail)

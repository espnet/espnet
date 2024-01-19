import json
import os
import sys

all_rare_words = set()
dirname = sys.argv[1]

with open("data/Blist/all_rare_words.txt") as fin:
    for line in fin:
        all_rare_words.add(line.strip())

perutt_blist = {}
with open(os.path.join(dirname, "text")) as fin:
    for line in fin:
        uttname = line.split()[0]
        content = line.split()[1:]
        perutt_blist[uttname] = []
        for word in content:
            if word in all_rare_words and word not in perutt_blist[uttname]:
                perutt_blist[uttname].append(word)

with open(os.path.join(dirname, "perutt_blist.json"), "w") as fout:
    json.dump(perutt_blist, fout, indent=4)

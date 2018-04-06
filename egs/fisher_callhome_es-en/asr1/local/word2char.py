#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import codecs

SPACE = '<space>'
SYM1 = '[laughter]'
SYM1_REP = '<laughter>'
SYM2 = '[noise]'
SYM2_REP = '<noise>'


def process_word(w):
    if w == SYM1:
        return SYM1_REP
    if w == SYM2:
        return SYM2_REP
    return ' '.join([c for c in w if c.isalnum()])

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")

    opt.add_argument('-t', action='store', dest='text', required=True)
    options = opt.parse_args()
    t_dict = {}
    new_text = codecs.open(options.text + '.char', 'w', 'utf-8')
    for line in codecs.open(options.text, 'r', 'utf-8').readlines():
        idx, line = line.strip().split(None, 1)
        new_line = ' <space> '.join([process_word(w) for w in line.strip().split()])
        for w in new_line.split():
            t_dict[w] = t_dict.get(w, len(t_dict))
        new_text.write(idx + ' ' + new_line + '\n')
    new_text.close()
    new_dict = codecs.open(options.text + '.char.dict', 'w', 'utf-8')
    for i in t_dict:
        new_dict.write(str(t_dict[i]) + ' ' + i + '\n')
    new_dict.close()

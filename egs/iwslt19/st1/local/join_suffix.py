#!/usr/bin/env python3
#
# Copyright  2014  Nickolay V. Shmyrev
#            2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0


from __future__ import print_function
import sys
from codecs import open

# This script joins together pairs of split-up words like "you 're" -> "you're".
# The TEDLIUM transcripts are normalized in a way that's not traditional for
# speech recognition.

prev_line = ''
for line in sys.stdin:
    if line == prev_line:
        continue
    items = line.split()
    new_items = []
    i = 0
    while i < len(items):
        if i < len(items) - 1 and items[i + 1][0] == '\'':
            new_items.append(items[i] + items[i + 1])
            i = i + 1
        else:
            new_items.append(items[i])
        i = i + 1
    print(' '.join(new_items))
    prev_line = line

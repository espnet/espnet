#!/usr/bin/env python3
import sys

for line in sys.stdin:
    seps = line.rstrip().split()
    uttid = seps[-1]
    assert uttid[0] == "("
    assert uttid[-1] == ")"
    print(uttid[1:-1], " ", *seps[:-1])

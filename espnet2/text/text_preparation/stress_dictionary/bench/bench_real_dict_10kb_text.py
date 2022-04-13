#!/usr/bin/env python3
import pyperf
import sys
from os.path import *
sys.path.append(dirname(dirname(abspath(__file__))))

import stresses

s = stresses.Stresses(['wiki-stresses.json'])

with open('bench/bench_text_10k.txt', 'r') as f:
    txt = f.read()

runner = pyperf.Runner()
runner.bench_func('Stresses.add()', lambda : s.add(txt))

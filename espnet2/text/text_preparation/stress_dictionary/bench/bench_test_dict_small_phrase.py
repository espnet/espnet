#!/usr/bin/env python3
import pyperf
import sys
from os.path import *
sys.path.append(dirname(dirname(abspath(__file__))))

import stresses

s = stresses.Stresses(['test/test_dict.json'])

runner = pyperf.Runner()
runner.bench_func('Stresses.add()', lambda : s.add('Мама мыла раму'))

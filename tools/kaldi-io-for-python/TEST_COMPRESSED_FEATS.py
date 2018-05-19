#!/bin/env python

import numpy as np
import kaldi_io
import timeit

t_beg = timeit.default_timer()
orig = {k:m for k,m in kaldi_io.read_mat_ark('data/feats.ark')}
print(timeit.default_timer() - t_beg);

t_beg = timeit.default_timer()
comp = {k:m for k,m in kaldi_io.read_mat_ark('data/feats_compressed.ark')}
print(timeit.default_timer() - t_beg);
# ~8-10x slower, this is already reasonable,

for key in orig.keys():
  print(key, np.sum(np.abs(comp[key]-orig[key])))
# => The values are not identical, but very similar.
#    Can it be the `order' of arithmetic operations?


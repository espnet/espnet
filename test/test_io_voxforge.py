# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import os

import numpy
import pytest


# TODO(karita): use much smaller corpus like AN4 and download if it does not exists
def test_voxforge_feats():
    import kaldi_io_py
    pytest.importorskip("kaldi_io")
    import kaldi_io

    train_scp = "scp:egs/voxforge/asr1/data/tr_it/feats.scp"
    if not os.path.exists(train_scp):
        pytest.skip("voxforge scp has not been created")

    r1 = kaldi_io_py.read_mat_scp(train_scp)
    r2 = kaldi_io.RandomAccessBaseFloatMatrixReader(train_scp)

    for k, v1 in r1:
        k = str(k)
        print(k)
        v2 = r2[k]
        assert v1.shape == v2.shape
        numpy.testing.assert_allclose(v1, v2, atol=1e-5)

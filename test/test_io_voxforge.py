# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import os
import sys

import pytest
import numpy


# TODO: use much smaller corpus like AN4 and download if it does not exists
def test_voxforge_feats():
    import kaldi_io_py
    import lazy_io
    try:
        import kaldi_io
    except:
        pytest.skip("kaldi_io (kaldi-python) is not installed")


    train_scp = "scp:egs/voxforge/asr1/data/tr_it/feats.scp"
    if not os.path.exists(train_scp):
        pytest.skip("voxforge scp has not been created")

    r1 = kaldi_io_py.read_mat_scp(train_scp)
    r2 = kaldi_io.RandomAccessBaseFloatMatrixReader(train_scp)
    r3 = lazy_io.read_dict_scp(train_scp)

    for k, v1 in r1:
        k = str(k)
        print(k)
        v2 = r2[k]
        v3 = r3[k]
        assert v1.shape == v2.shape
        assert v1.shape == v3.shape
        numpy.testing.assert_allclose(v1, v2, atol=1e-5)
        numpy.testing.assert_allclose(v1, v3, atol=0)


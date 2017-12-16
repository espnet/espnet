# coding: utf-8
import numpy
import kaldi_io
import kaldi_io_py
import lazy_io

train_scp = "scp:egs/voxforge/asr1/data/tr_it/feats.scp"

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


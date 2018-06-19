# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import pytest
pytest.importorskip('torch')
import torch  # NOQA
from e2e_asr_attctc_th import pad_list, mask_by_length  # NOQA


def test_pad_list():
    xs = [[1, 2, 3],
          [1, 2],
          [1, 2, 3, 4]]
    xs = list(map(lambda x: torch.LongTensor(x), xs))
    xpad = pad_list(xs, -1)

    es = [[1, 2, 3, -1],
          [1, 2, -1, -1],
          [1, 2, 3, 4]]
    assert xpad.data.tolist() == es


def test_mask_by_length():
    xs = [[1, 2, 3, -1],
          [1, 2, -1, -1],
          [1, 2, 3, 4]]
    xs = torch.LongTensor(xs)
    xlen = [3, 2, 4]
    ys = mask_by_length(xs, xlen, fill=0)
    es = [[1, 2, 3, 0],
          [1, 2, 0, 0],
          [1, 2, 3, 4]]
    assert ys.data.tolist() == es

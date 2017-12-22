import pytest
pytest.importorskip('torch')
import torch
from torch.autograd import Variable
from e2e_asr_attctc_th import pad_list, mask_by_length


def test_pad_list():
    xs = [[1, 2, 3],
          [1, 2],
          [1, 2, 3, 4]]
    xs = list(map(lambda x: Variable(torch.LongTensor(x)), xs))
    xpad = pad_list(xs, -1)
    
    es = [[1, 2, 3, -1],
          [1, 2, -1, -1],
          [1, 2, 3, 4]]
    assert xpad.data.tolist() == es


def test_mask_by_length():
    xs = [[1, 2, 3, -1],
          [1, 2, -1, -1],
          [1, 2, 3, 4]]
    xs = Variable(torch.LongTensor(xs))
    xlen = [3, 2, 4]
    ys = mask_by_length(xs, xlen, fill=0)
    es = [[1, 2, 3, 0],
          [1, 2, 0, 0],
          [1, 2, 3, 4]]
    assert ys.data.tolist() == es

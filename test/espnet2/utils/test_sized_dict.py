import multiprocessing
import sys

import numpy as np
import pytest
import torch.multiprocessing

from espnet2.utils.sized_dict import SizedDict, get_size


def test_get_size():
    d = {}
    x = np.random.randn(10)
    d["a"] = x
    size1 = sys.getsizeof(d)
    assert size1 + get_size(x) + get_size("a") == get_size(d)


def test_SizedDict_size():
    d = SizedDict()
    assert d.size == 0

    x = np.random.randn(10)
    d["a"] = x
    assert d.size == get_size(x) + sys.getsizeof("a")

    y = np.random.randn(10)
    d["b"] = y
    assert d.size == get_size(x) + get_size(y) + sys.getsizeof("a") + sys.getsizeof("b")

    # Overwrite
    z = np.random.randn(10)
    d["b"] = z
    assert d.size == get_size(x) + get_size(z) + sys.getsizeof("a") + sys.getsizeof("b")


def _set(d):
    d["a"][0] = 10


@pytest.mark.execution_timeout(5)
def test_SizedDict_shared():
    d = SizedDict(shared=True)
    x = torch.randn(10)
    d["a"] = x

    mp = multiprocessing.get_context("forkserver")
    p = mp.Process(target=_set, args=(d,))
    p.start()
    p.join()
    assert d["a"][0] == 10


def test_SizedDict_getitem():
    d = SizedDict(data={"a": 2, "b": 5, "c": 10})
    assert d["a"] == 2


def test_SizedDict_iter():
    d = SizedDict(data={"a": 2, "b": 5, "c": 10})
    assert list(iter(d)) == ["a", "b", "c"]


def test_SizedDict_contains():
    d = SizedDict(data={"a": 2, "b": 5, "c": 10})
    assert "a" in d


def test_SizedDict_len():
    d = SizedDict(data={"a": 2, "b": 5, "c": 10})
    assert len(d) == 3

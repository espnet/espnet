from collections import deque
from test.memory_growth import assert_memory_bounded, retained_nbytes

import numpy as np
import pytest
import torch


class _Leaky:
    """Append every chunk and never drop one."""

    def __init__(self):
        self.history = []

    def step(self):
        self.history.append(np.zeros(1000, dtype=np.float32))


class _Bounded:
    """Keep the last 10 chunks."""

    def __init__(self):
        self.history = deque(maxlen=10)

    def step(self):
        self.history.append(np.zeros(1000, dtype=np.float32))


def test_retained_nbytes_counts_nested_data():
    obj = _Bounded()
    obj.step()
    obj.extra = {"t": torch.zeros(250), "s": [np.zeros(10, dtype=np.int16)]}
    n = retained_nbytes(obj)
    assert n >= 4000 + 1000 + 20


def test_retained_nbytes_counts_the_base_of_a_view():
    a = np.zeros(1000, dtype=np.float32)
    holder = type("Holder", (), {})()
    holder.view = a[:10]
    assert retained_nbytes(holder) >= 4000


def test_a_leak_fails():
    obj = _Leaky()
    with pytest.raises(AssertionError, match=r"grown most: history \+"):
        assert_memory_bounded(obj, obj.step, steps=200)


def test_a_leak_through_a_view_fails():
    # Each call keeps a 10-sample view of an array that grows by 1000.
    obj = type("Holder", (), {})()
    obj.all = np.zeros(0, dtype=np.float32)

    def step():
        obj.all = np.concatenate([obj.all, np.zeros(1000, dtype=np.float32)])
        obj.tail = obj.all[-10:]

    with pytest.raises(AssertionError, match="grows with the stream"):
        assert_memory_bounded(obj, step, steps=200)


def test_bounded_memory_passes():
    obj = _Bounded()
    sizes = assert_memory_bounded(obj, obj.step, steps=200)
    assert len(sizes) == 20


def test_memory_that_varies_within_a_bound_passes():
    """A buffer that fills and empties, as one turn after another does."""
    obj = type("Holder", (), {})()
    obj.buf = np.zeros(0, dtype=np.float32)
    n = [0]

    def step():
        n[0] += 1
        if n[0] % 7 == 0:
            obj.buf = np.zeros(0, dtype=np.float32)
        else:
            obj.buf = np.concatenate([obj.buf, np.zeros(1000, dtype=np.float32)])

    assert_memory_bounded(obj, step, steps=500)

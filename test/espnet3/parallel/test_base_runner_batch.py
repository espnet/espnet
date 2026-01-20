import pytest

from espnet3.parallel.base_runner import BaseRunner


class DummyProvider:
    def build_env_local(self):
        return {"dataset": None, "model": None}


class DummyRunner(BaseRunner):
    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return idx


def test_batch_size_chunks_indices():
    runner = DummyRunner(DummyProvider(), batch_size=2)
    out = runner([0, 1, 2, 3, 4])

    assert out == [[0, 1], [2, 3], [4]]


def test_batch_size_rejects_non_positive():
    runner = DummyRunner(DummyProvider(), batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        runner([0])

def test_batch_size_rejects_None():
    runner = DummyRunner(DummyProvider(), batch_size=None)
    
    # No error should be raised here
    out = runner([0])

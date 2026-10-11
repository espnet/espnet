import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.metrics import (
    MetricContractError,
    check_metric_batch,
)
from espnet3.components.metrics.base_metric import BaseMetric


class Pair(BaseMetric):
    inputs = (
        Field("ref", "text"),
        Field("hyp", "text"),
        Field("prompt", "text", optional=True),
    )
    outputs = (Field("n", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("n", 0, "sum")

    def update(self, *, ref, hyp, prompt=None):
        self.n += len(ref)

    def compute(self):
        return {"n": self.n}


class Speech(BaseMetric):
    inputs = (Field("speech", "audio"),)
    outputs = (Field("n", "number"),)

    def update(self, *, speech):
        pass

    def compute(self):
        return {"n": 0}


def test_a_good_text_batch_updates():
    m = Pair()
    m.update(ref=["a", "b"], hyp=["a", "c"])
    assert m.n == 2


def test_a_tuple_batch_passes():
    check_metric_batch(Pair(), {"ref": ("a",), "hyp": ("b",)})


def test_an_absent_or_none_optional_input_passes():
    m = Pair()
    m.update(ref=["a"], hyp=["b"])
    m.update(ref=["a"], hyp=["b"], prompt=None)
    assert m.n == 2


def test_a_good_audio_batch_passes():
    check_metric_batch(Speech(), {"speech": [np.zeros(16, dtype=np.float32), "a.wav"]})


def test_an_unknown_input_raises():
    with pytest.raises(MetricContractError, match="got input 'foo'"):
        check_metric_batch(Pair(), values={"ref": ["a"], "hyp": ["b"], "foo": ["c"]})


def test_a_missing_input_raises():
    with pytest.raises(MetricContractError, match="input 'hyp' is missing"):
        check_metric_batch(Pair(), values={"ref": ["a"]})


def test_a_value_that_is_not_a_sequence_raises():
    with pytest.raises(MetricContractError, match="must be a list or tuple, not str"):
        check_metric_batch(Pair(), values={"ref": ["a"], "hyp": "b"})


def test_unequal_lengths_raise():
    with pytest.raises(MetricContractError, match="'hyp' has 1 items, 'ref' has 2"):
        check_metric_batch(Pair(), values={"ref": ["a", "b"], "hyp": ["a"]})


def test_a_wrong_kind_names_the_position():
    with pytest.raises(MetricContractError, match=r"'hyp'\[1\] is int, must be text"):
        check_metric_batch(Pair(), values={"ref": ["a", "b"], "hyp": ["a", 3]})


def test_update_runs_the_check():
    with pytest.raises(MetricContractError):
        Pair().update(ref=["a"], hyp=[3])

import pytest

from espnet3.parallel.base_runner import BaseRunner
from espnet3.systems.base.inference_runner import InferenceRunner


class DummyProvider:
    pass


class DummyRunner(InferenceRunner):
    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return {"idx": idx, "hyp": "h", "ref": "r"}


def test_validate_output_accepts_expected_keys():
    runner = DummyRunner(
        DummyProvider(), idx_key="idx", hyp_key=["hyp", "alt"], ref_key="ref"
    )
    output = {"idx": 1, "hyp": "h", "alt": "a", "ref": "r"}

    runner._validate_output(output)


def test_validate_output_rejects_missing_or_extra_keys():
    runner = DummyRunner(DummyProvider())
    with pytest.raises(ValueError, match="missing=\\['ref'\\]"):
        runner._validate_output({"idx": 1, "hyp": "h"})
    runner._validate_output({"idx": 1, "hyp": "h", "ref": "r", "extra": 0})


def test_validate_output_rejects_non_dict_and_idx_list():
    runner = DummyRunner(DummyProvider())
    with pytest.raises(TypeError, match="Expected dict output"):
        runner._validate_output("bad")
    with pytest.raises(TypeError, match="'idx' must be a scalar"):
        runner._validate_output({"idx": [1], "hyp": "h", "ref": "r"})


def test_call_async_returns_raw(monkeypatch):
    def fake_base_call(self, indices):
        return ["raw"]

    monkeypatch.setattr(BaseRunner, "__call__", fake_base_call)
    runner = DummyRunner(DummyProvider(), async_mode=True)

    assert runner([0]) == ["raw"]


def test_call_flattens_and_validates(monkeypatch):
    def fake_base_call(self, indices):
        return [
            {"idx": 0, "hyp": "h0", "ref": "r0"},
            [{"idx": 1, "hyp": "h1", "ref": "r1"}],
        ]

    monkeypatch.setattr(BaseRunner, "__call__", fake_base_call)
    runner = DummyRunner(DummyProvider())

    assert runner([0, 1]) == [
        {"idx": 0, "hyp": "h0", "ref": "r0"},
        {"idx": 1, "hyp": "h1", "ref": "r1"},
    ]


def test_call_returns_none(monkeypatch):
    monkeypatch.setattr(BaseRunner, "__call__", lambda self, indices: None)
    runner = DummyRunner(DummyProvider())

    assert runner([0]) is None


def test_call_propagates_validation_error(monkeypatch):
    monkeypatch.setattr(
        BaseRunner, "__call__", lambda self, indices: [{"idx": 0, "hyp": "h"}]
    )
    runner = DummyRunner(DummyProvider())

    with pytest.raises(ValueError, match="missing=\\['ref'\\]"):
        runner([0])

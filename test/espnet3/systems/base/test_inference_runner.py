import pytest

from espnet3.parallel.base_runner import BaseRunner
from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn


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


def test_resolve_idx_key_uses_default_utt_id():
    runner = DummyRunner(DummyProvider())

    assert runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"}) == "utt_id"


def test_resolve_idx_key_uses_explicit_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    assert (
        runner.resolve_idx_key({"sample_id": "u1", "hyp": "h", "ref": "r"})
        == "sample_id"
    )


def test_resolve_idx_key_rejects_missing_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    with pytest.raises(ValueError, match="idx_key='sample_id'"):
        runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"})


def test_validate_output_rejects_missing_or_extra_keys():
    runner = DummyRunner(DummyProvider())
    with pytest.raises(ValueError, match="idx_key='utt_id'"):
        runner._validate_output({"hyp": "h"})
    runner._validate_output({"utt_id": "u1", "hyp": "h", "ref": "r", "extra": 0})


def test_validate_output_rejects_non_dict_and_idx_list():
    runner = DummyRunner(DummyProvider())
    with pytest.raises(TypeError, match="Expected dict output"):
        runner._validate_output("bad")
    with pytest.raises(TypeError, match="'utt_id' must be a single value"):
        runner._validate_output({"utt_id": [1], "hyp": "h", "ref": "r"})


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
    runner = DummyRunner(DummyProvider(), idx_key="idx")

    assert runner([0, 1]) == [
        {"idx": 0, "hyp": "h0", "ref": "r0"},
        {"idx": 1, "hyp": "h1", "ref": "r1"},
    ]


def test_call_returns_none(monkeypatch):
    monkeypatch.setattr(BaseRunner, "__call__", lambda self, indices: None)
    runner = DummyRunner(DummyProvider())

    assert runner([0]) is None


def test_call_propagates_validation_error(monkeypatch):
    monkeypatch.setattr(BaseRunner, "__call__", lambda self, indices: [{"hyp": "h"}])
    runner = DummyRunner(DummyProvider())

    with pytest.raises(ValueError, match="idx_key='utt_id'"):
        runner([0])


def test_load_output_fn_rejects_missing_module():
    with pytest.raises(ModuleNotFoundError):
        _load_output_fn("no.such.module.output_fn")


def test_validate_output_rejects_missing_hyp_ref_keys():
    runner = DummyRunner(
        DummyProvider(), idx_key="utt_id", hyp_key="hyp", ref_key="ref"
    )
    with pytest.raises(ValueError, match="missing="):
        runner._validate_output({"utt_id": "u1"})


def test_forward_raises_without_input_key_kwarg():
    with pytest.raises(RuntimeError, match="input_key must be provided"):
        InferenceRunner.forward(0, dataset=[], model=lambda: None)


def test_forward_single_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            0, dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        [0, 1], dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": [1.0, 2.0]}


def test_forward_single_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        0, dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": 1.0}


def test_forward_batched_wraps_model_exception_in_runtime_error():
    dataset = [{"speech": 1.0}]

    def failing_model(speech):
        raise ValueError("model broken")

    with pytest.raises(RuntimeError, match="Batched inference failed"):
        InferenceRunner.forward(
            [0], dataset=dataset, model=failing_model, input_key="speech"
        )

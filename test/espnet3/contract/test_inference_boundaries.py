"""Failure boundaries of config-declared inference contracts."""

import copy

import numpy as np
import pytest

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.normalizers import NORMALIZERS
from espnet3.systems.base.inference_runner import InferenceRunner


def _config():
    return {
        "inputs": {"text": {"kind": "text"}},
        "outputs": {"text": {"kind": "text"}},
    }


@pytest.mark.parametrize(
    "change,error",
    [
        ({"inputs": {}}, ValueError),
        ({"outputs": {}}, ValueError),
        ({"inputs": []}, ValueError),
        ({"outputs": {"text": 1}}, TypeError),
        ({"outputs": {"text": {"kind": "text", "optional": True}}}, ValueError),
        ({"inputs": {"text": {"kind": "missing_kind"}}}, ValueError),
        ({"inputs": {"text": {"kind": "text", "bogus": 1}}}, ValueError),
        ({"inputs": {"text": {"kind": "text", "normalize": ["missing"]}}}, ValueError),
        ({"imports": "some_module"}, TypeError),
        ({"imports": [None]}, TypeError),
        ({"unknown": True}, ValueError),
    ],
)
def test_invalid_contract_configuration(change, error):
    """Malformed declarations fail at contract loading."""
    config = _config()
    config.update(change)
    with pytest.raises(error):
        load_inference_contract(config)


@pytest.mark.parametrize("keys", [None, [], ["other"], ["text", "text"]])
def test_input_key_must_match_contract(keys):
    """Missing, different and duplicate input keys are rejected."""
    with pytest.raises(ValueError, match="input_key"):
        load_inference_contract(_config()).validate_input_keys(keys)


@pytest.mark.parametrize("sample", [{}, {"text": None}, {"text": 1}])
@pytest.mark.parametrize("batched", [False, True])
def test_invalid_input_prevents_backend_call(sample, batched):
    """One invalid item prevents the entire current batch from decoding."""
    calls = []
    dataset = [{"text": "good"}, sample] if batched else [sample]
    original = copy.deepcopy(dataset)
    with pytest.raises((KeyError, TypeError)):
        InferenceRunner.forward(
            [0, 1] if batched else 0,
            dataset=dataset,
            model=lambda **kwargs: calls.append(kwargs),
            input_key="text",
            inference_contract=_config(),
        )
    assert calls == []
    assert dataset == original


@pytest.mark.parametrize(
    "output,error",
    [
        ({}, KeyError),
        ({"text": "ok", "extra": "bad"}, ValueError),
        ({"text": 42}, TypeError),
        ({"text": None}, TypeError),
        ({"text": ["ok"]}, TypeError),
        ("not a mapping", TypeError),
    ],
)
def test_invalid_backend_output_fails_contract(output, error):
    """A malformed output is not silently coerced into a text result."""
    calls = []

    def backend(text):
        calls.append(text)
        return output

    with pytest.raises(error):
        InferenceRunner.forward(
            0,
            dataset=[{"text": "in"}],
            model=backend,
            input_key="text",
            inference_contract=_config(),
        )
    assert calls == ["in"]


@pytest.mark.parametrize("text", ["", "  ", "日本語。", "a" * 4096])
def test_valid_text_boundaries_are_preserved(text):
    """Empty/Unicode/long text remains text without implicit normalization."""
    result = InferenceRunner.forward(
        0,
        dataset=[{"text": text}],
        model=lambda text: {"text": text},
        input_key="text",
        inference_contract=_config(),
    )
    assert result == {"text": text, "utt_id": "0"}


@pytest.mark.parametrize(
    "output",
    [
        [],
        [{"text": "only one"}],
        [{"text": "a"}, {"text": "b"}, {"text": "extra"}],
        {"text": "one value for two samples"},
        {"text": ["a"]},
        [{"text": "a"}, {"text": 42}],
    ],
)
def test_malformed_batch_cannot_truncate_or_return_partial_success(output):
    """Exactly one valid mapping is required per input sample."""
    with pytest.raises((TypeError, ValueError)):
        InferenceRunner.forward(
            [0, 1],
            dataset=[{"text": "a"}, {"text": "b"}],
            model=lambda text: output,
            input_key="text",
            inference_contract=_config(),
        )


@pytest.mark.parametrize("batched", [False, True])
def test_normalizers_run_once_per_value_and_in_declared_order(monkeypatch, batched):
    """Non-idempotent normalizers expose duplicate boundary application."""
    events = []

    def append_a(value):
        events.append(("a", value))
        return value + "a"

    def append_b(value):
        events.append(("b", value))
        return value + "b"

    monkeypatch.setitem(NORMALIZERS, "boundary_a", append_a)
    monkeypatch.setitem(NORMALIZERS, "boundary_b", append_b)
    config = _config()
    config["inputs"]["text"]["normalize"] = ["boundary_a", "boundary_b"]
    config["outputs"]["text"]["normalize"] = ["boundary_b", "boundary_a"]
    calls = []

    def backend(text):
        calls.append(text)
        if isinstance(text, list):
            return [{"text": value} for value in text]
        return {"text": text}

    result = InferenceRunner.forward(
        [0, 1] if batched else 0,
        dataset=[{"text": "x"}, {"text": "y"}],
        model=backend,
        input_key="text",
        inference_contract=config,
    )
    assert calls == ([["xab", "yab"]] if batched else ["xab"])
    assert events == (
        [
            ("a", "x"),
            ("b", "xa"),
            ("a", "y"),
            ("b", "ya"),
            ("b", "xab"),
            ("a", "xabb"),
            ("b", "yab"),
            ("a", "yabb"),
        ]
        if batched
        else [("a", "x"), ("b", "xa"), ("b", "xab"), ("a", "xabb")]
    )
    assert result == (
        [{"text": "xabba", "utt_id": "0"}, {"text": "yabba", "utt_id": "1"}]
        if batched
        else {"text": "xabba", "utt_id": "0"}
    )


@pytest.mark.parametrize("side", ["inputs", "outputs"])
@pytest.mark.parametrize("raises", [False, True])
def test_normalizer_failure_is_checked_at_its_boundary(monkeypatch, side, raises):
    """Exceptions and wrong-kind normalized values cannot become success."""
    calls = []

    def normalize(value):
        if raises:
            raise RuntimeError("normalizer failed")
        return 123

    monkeypatch.setitem(NORMALIZERS, "boundary_failure", normalize)
    config = _config()
    config[side]["text"]["normalize"] = ["boundary_failure"]

    def backend(text):
        calls.append(text)
        return {"text": text}

    with pytest.raises(RuntimeError if raises else TypeError):
        InferenceRunner.forward(
            0,
            dataset=[{"text": "hello"}],
            model=backend,
            input_key="text",
            inference_contract=config,
        )
    assert calls == ([] if side == "inputs" else ["hello"])


@pytest.mark.parametrize("rate", [0, -1, True, 16000.5])
def test_audio_contract_rejects_invalid_rate_declaration(rate):
    """The configured rate must be a positive integer, not bool/float."""
    config = _config()
    config["inputs"] = {"speech": {"kind": "audio", "sample_rate": rate}}
    with pytest.raises(ValueError, match="sample_rate"):
        load_inference_contract(config)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_fails_before_decoding(bad):
    """Nonfinite sample rejection is shared by every contracted runner."""
    calls = []
    with pytest.raises(ValueError, match="finite"):
        InferenceRunner.forward(
            0,
            dataset=[{"speech": (16000, np.array([0, bad, 0]))}],
            model=lambda **kwargs: calls.append(kwargs),
            input_key="speech",
            inference_contract={
                "inputs": {"speech": {"kind": "audio", "sample_rate": 16000}},
                "outputs": {"text": {"kind": "text"}},
            },
        )
    assert calls == []


@pytest.mark.parametrize("optional", [None, "context", "absent"])
@pytest.mark.parametrize("batched", [False, True])
def test_optional_input_absent_none_and_present(optional, batched):
    config = _config()
    config["inputs"]["context"] = {"kind": "text", "optional": True}
    sample = {"text": "hello", "reference": "dataset metadata is allowed"}
    if optional != "absent":
        sample["context"] = optional
    seen = []

    def backend(text, **kwargs):
        seen.append(kwargs)
        return {"text": text}

    result = InferenceRunner.forward(
        [0, 1] if batched else 0,
        dataset=[sample, sample],
        model=backend,
        input_key=["text", "context"],
        inference_contract=config,
    )
    assert seen == (
        [{"context": ["context", "context"] if batched else "context"}]
        if optional == "context"
        else [{}]
    )
    assert isinstance(result, list if batched else dict)


def test_reserved_output_identifier_cannot_replace_sample_identifier():
    config = _config()
    config["outputs"]["utt_id"] = {"kind": "text"}
    with pytest.raises(ValueError, match="utt_id"):
        InferenceRunner.forward(
            0,
            dataset=[{"text": "hello", "utt_id": "real"}],
            model=lambda text: {"text": text, "utt_id": "forged"},
            input_key="text",
            inference_contract=config,
        )

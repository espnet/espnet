"""Inference contract behavior shared by recipe, publication, and demo."""

import numpy as np
import pytest
import soundfile as sf

from espnet3.contract import KINDS, Kind
from espnet3.contract.inference import (
    load_inference_contract,
    prepare_inference_inputs,
    resolve_inference_contract,
    validate_inference_outputs,
)
from espnet3.contract.kinds.audio import Audio, AudioKind
from espnet3.contract.normalizers import NORMALIZERS, register_normalizer
from espnet3.systems.base.inference_runner import (
    InferenceRunner,
    _materialize_output_value,
)


def _contract():
    return load_inference_contract(
        {
            "inputs": {
                "speech": {
                    "kind": "audio",
                    "sample_rate": 16000,
                    "rate_policy": "resample",
                }
            },
            "outputs": {"text": {"kind": "text", "normalize": ["strip", "to_lower"]}},
        }
    )


def test_custom_kind_controls_options_and_model_input(monkeypatch):
    """New modalities define their own options without editing inference code."""

    class TokensKind(Kind):
        option_names = Kind.option_names | {"max_length"}

        def check(self, value, field, model, *, output):
            if not isinstance(value, list) or not all(
                isinstance(v, int) for v in value
            ):
                raise TypeError("tokens must be a list of integers")
            return value

        def check_configured(self, value, field, model, options, *, output):
            value = self.check(value, field, model, output=output)
            if len(value) > options["max_length"]:
                raise ValueError("too many tokens")
            return value

        def to_model_input(self, value, field):
            return tuple(value)

    monkeypatch.setitem(KINDS, "test_tokens", TokensKind())
    contract = load_inference_contract(
        {
            "inputs": {"tokens": {"kind": "test_tokens", "max_length": 2}},
            "outputs": {"tokens": {"kind": "test_tokens", "max_length": 2}},
        }
    )
    seen = []

    def model(tokens):
        seen.append(tokens)
        return {"tokens": list(tokens)}

    result = InferenceRunner.forward(
        0,
        dataset=[{"tokens": [1, 2]}],
        model=model,
        input_key="tokens",
        inference_contract=contract,
    )
    assert seen == [(1, 2)]
    assert result == {"tokens": [1, 2], "utt_id": "0"}
    with pytest.raises(ValueError, match="too many tokens"):
        contract.prepare_input({"tokens": [1, 2, 3]}, model=None)


def test_audio_kind_options_work_under_another_registered_name(monkeypatch):
    """Rate handling and array conversion belong to the Kind, not its name."""
    monkeypatch.setitem(KINDS, "test_waveform", AudioKind())
    contract = load_inference_contract(
        {
            "inputs": {"wave": {"kind": "test_waveform", "sample_rate": 16000}},
            "outputs": {"text": {"kind": "text"}},
        }
    )
    result = contract.prepare_input({"wave": (8000, np.ones(80))}, model=None)
    assert result["wave"].array.shape == (160,)


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "text", "sample_rate": 16000},
        {"kind": "audio", "normalize": ["to_lower"]},
    ],
)
def test_options_and_builtin_normalizers_are_owned_by_the_kind(options):
    """A Kind rejects options and builtin transforms belonging to another Kind."""
    with pytest.raises(ValueError, match="Unknown"):
        load_inference_contract(
            {"inputs": {"value": options}, "outputs": {"text": {"kind": "text"}}}
        )


def test_custom_runner_uses_contract_on_both_sides_of_forward():
    """Custom forward implementations still pass through both contract hooks."""
    seen = []

    class CustomRunner(InferenceRunner):
        @staticmethod
        def forward(idx, *, dataset, model, **kwargs):
            contract = resolve_inference_contract(kwargs["inference_contract"])
            contract.validate_input_keys(kwargs["input_key"])
            prepared, original = prepare_inference_inputs(dataset, idx, contract, model)
            sample = prepared[idx]
            seen.append(sample["speech"])
            result = model(speech=sample["speech"])
            return validate_inference_outputs(
                result, idx, contract, model, original, kwargs.get("idx_key", "utt_id")
            )

    source = np.ones(800, dtype=np.int16)
    sample = {"utt_id": "utt-1", "speech": (8000, source)}
    result = CustomRunner.forward(
        0,
        dataset=[sample],
        model=lambda speech: {"text": "  HELLO  "},
        input_key="speech",
        inference_contract=_contract(),
    )

    assert result == {"text": "hello", "utt_id": "utt-1"}
    assert seen[0].array.shape == (1600,)
    assert seen[0].array.dtype == np.float32
    assert sample["speech"][0] == 8000


def test_reject_audio_rate_before_calling_backend():
    """A rate mismatch is reported before a model call is attempted."""
    config = {
        "inputs": {
            "speech": {
                "kind": "audio",
                "sample_rate": 16000,
                "rate_policy": "reject",
            }
        },
        "outputs": {"text": {"kind": "text"}},
    }
    called = []

    def backend(speech):
        called.append(speech)
        return {"text": "hello"}

    with pytest.raises(ValueError, match="sample rate 44100"):
        InferenceRunner.forward(
            0,
            dataset=[{"speech": (44100, np.zeros(100))}],
            model=backend,
            input_key="speech",
            inference_contract=config,
        )
    assert called == []


def test_reject_audio_rate_from_file_path(tmp_path):
    """File inputs are checked against their header rate in reject mode."""
    path = tmp_path / "wrong-rate.wav"
    sf.write(path, np.zeros(441, dtype=np.float32), 44100)
    contract = load_inference_contract(
        {
            "inputs": {
                "speech": {
                    "kind": "audio",
                    "sample_rate": 16000,
                    "rate_policy": "reject",
                }
            },
            "outputs": {"text": {"kind": "text"}},
        }
    )

    with pytest.raises(ValueError, match="sample rate 44100"):
        contract.prepare_input({"speech": path}, model=None)


def test_batch_failure_propagates_without_retry():
    """A rejected batch propagates and is not retried item by item."""
    normalized = []

    def strip_once(text):
        normalized.append(text)
        return text.strip()

    register_normalizer("test_strip_once", strip_once, replace=True)
    contract = load_inference_contract(
        {
            "inputs": {"text": {"kind": "text", "normalize": ["test_strip_once"]}},
            "outputs": {"text": {"kind": "text", "normalize": ["to_lower"]}},
        }
    )
    calls = []

    def backend(text):
        calls.append(text)
        if isinstance(text, list):
            raise TypeError("batch unsupported")
        return {"text": text}

    with pytest.raises(TypeError, match="batch unsupported"):
        InferenceRunner.forward(
            [0, 1],
            dataset=[{"text": " A "}, {"text": " B "}],
            model=backend,
            input_key="text",
            inference_contract=contract,
        )
    assert calls == [["A", "B"]]
    assert normalized == [" A ", " B "]


def test_batched_mapping_of_field_lists_is_split_per_sample():
    """Backends that return one list per field can use the shared contract."""
    contract = load_inference_contract(
        {
            "inputs": {"text": {"kind": "text"}},
            "outputs": {"text": {"kind": "text", "normalize": ["to_lower"]}},
        }
    )

    result = InferenceRunner.forward(
        [0, 1],
        dataset=[{"text": "A"}, {"text": "B"}],
        model=lambda text: {"text": text},
        input_key="text",
        inference_contract=contract,
    )

    assert result == [
        {"text": "a", "utt_id": "0"},
        {"text": "b", "utt_id": "1"},
    ]


@pytest.mark.parametrize(("channels", "shape"), [(1, (160,)), (None, (2, 160))])
def test_contracted_audio_output_is_written_with_its_rate(tmp_path, channels, shape):
    """An audio Kind result carries its sample rate to the WAV writer."""
    contract = load_inference_contract(
        {
            "inputs": {"text": {"kind": "text"}},
            "outputs": {
                "audio": {
                    "kind": "audio",
                    "sample_rate": 16000,
                    "channels": channels,
                }
            },
        }
    )
    result = contract.prepare_output(
        {"audio": np.zeros(shape, dtype=np.float32)},
        model=None,
        idx=0,
        idx_key="utt_id",
        sample={"text": "hello"},
    )
    path = _materialize_output_value(
        result["utt_id"], "audio", result["audio"], tmp_path, None
    )

    assert sf.info(path).samplerate == 16000
    assert sf.info(path).channels == (channels or 2)


def test_registered_audio_normalizer_can_reuse_a_recipe_transform():
    """A recipe may register one waveform transform for contract processing."""

    def scale_audio(audio):
        return Audio(audio.array * 0.5, audio.rate)

    register_normalizer("test_scale_audio_half", scale_audio, replace=True)
    contract = load_inference_contract(
        {
            "inputs": {
                "speech": {
                    "kind": "audio",
                    "sample_rate": 16000,
                    "normalize": ["test_scale_audio_half"],
                }
            },
            "outputs": {"text": {"kind": "text"}},
        }
    )

    prepared = contract.prepare_input(
        {"speech": np.ones(10, dtype=np.float32)}, model=None
    )

    np.testing.assert_allclose(prepared["speech"].array, 0.5)


def test_contract_imports_recipe_module_in_worker_context(tmp_path, monkeypatch):
    """Worker calls load recipe registrations using the configured recipe root."""
    import sys

    module_name = "inference_contract_worker_plugin"
    (tmp_path / f"{module_name}.py").write_text(
        "from espnet3.contract import register_normalizer\n"
        "register_normalizer('worker_suffix', lambda text: text + '!')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])

    result = InferenceRunner.forward(
        0,
        dataset=[{"text": "hello"}],
        model=lambda text: {"text": text},
        input_key="text",
        inference_contract={
            "imports": [module_name],
            "inputs": {"text": {"kind": "text"}},
            "outputs": {"text": {"kind": "text", "normalize": ["worker_suffix"]}},
        },
        inference_contract_root=str(tmp_path),
    )

    assert result["text"] == "hello!"


def test_contract_rejects_legacy_output_fn():
    """A contracted backend must return named outputs itself."""
    contract = load_inference_contract(
        {
            "inputs": {"text": {"kind": "text"}},
            "outputs": {"text": {"kind": "text"}},
        }
    )

    with pytest.raises(ValueError, match="output_fn cannot be combined"):
        InferenceRunner.forward(
            0,
            dataset=[{"text": "hello"}],
            model=lambda text: {"text": text},
            input_key="text",
            inference_contract=contract,
            output_fn=lambda **kwargs: kwargs,
        )


@pytest.mark.parametrize("rate", [8000, 16000])
def test_audio_preserves_rate_and_all_channels_without_a_fixed_rate(rate):
    """Variable-rate models receive Audio with every channel and its rate."""
    contract = load_inference_contract(
        {
            "inputs": {"speech": {"kind": "audio"}},
            "outputs": {"speech": {"kind": "audio"}},
        }
    )
    source = np.stack([np.full(160, 0.25), np.full(160, 0.75)])
    seen = []

    def backend(speech):
        seen.append(speech)
        return {"speech": speech}

    result = InferenceRunner.forward(
        0,
        dataset=[{"speech": (rate, source)}],
        model=backend,
        input_key="speech",
        inference_contract=contract,
    )
    assert isinstance(seen[0], Audio)
    assert seen[0].rate == rate
    np.testing.assert_array_equal(seen[0].array, source)
    np.testing.assert_array_equal(result["speech"].array, source)
    assert result["speech"].rate == rate


@pytest.mark.parametrize(
    ("normalizer", "expected"),
    [
        ("first_channel", 0.25),
        ("mean_channels", 0.5),
    ],
)
def test_audio_channel_reduction_is_explicit(normalizer, expected):
    """Channel selection or averaging happens only when requested in config."""
    contract = load_inference_contract(
        {
            "inputs": {"speech": {"kind": "audio", "normalize": [normalizer]}},
            "outputs": {"text": {"kind": "text"}},
        }
    )
    source = np.stack([np.full(160, 0.25), np.full(160, 0.75)])
    audio = contract.prepare_input({"speech": (16000, source)}, model=None)["speech"]
    assert audio.rate == 16000
    assert audio.array.shape == (160,)
    np.testing.assert_allclose(audio.array, expected)
    np.testing.assert_allclose(source[1], 0.75)


def _stereo():
    return np.stack([np.full(160, 0.25), np.full(160, 0.75)])


def _mono():
    return np.full(160, 0.25)


def _channels_seen_by_model(spec, source, batched):
    """Run the runner and return the channel count of each audio the model got."""
    contract = load_inference_contract(
        {"inputs": {"speech": spec}, "outputs": {"text": {"kind": "text"}}}
    )
    seen = []

    def backend(speech):
        seen.append(speech)
        items = speech if batched else [speech]
        return [{"text": "x"} for _ in items] if batched else {"text": "x"}

    sample = {"speech": (16000, source)}
    InferenceRunner.forward(
        [0, 1] if batched else 0,
        dataset=[sample, sample],
        model=backend,
        input_key="speech",
        inference_contract=contract,
    )
    items = seen[0] if batched else [seen[0]]
    assert all(isinstance(item, Audio) for item in items)
    return [item.channels for item in items]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    ("spec", "source", "expected"),
    [
        # No channels declared: whatever the input has reaches the model.
        ({"kind": "audio"}, _stereo, 2),
        ({"kind": "audio"}, _mono, 1),
        # channels declared: the model gets exactly that many.
        ({"kind": "audio", "channels": 2}, _stereo, 2),
        ({"kind": "audio", "channels": 1}, _mono, 1),
        # Normalizers run before the channel check, so a reduction satisfies it.
        ({"kind": "audio", "normalize": ["mean_channels"]}, _stereo, 1),
        ({"kind": "audio", "normalize": ["first_channel"]}, _stereo, 1),
        (
            {"kind": "audio", "channels": 1, "normalize": ["mean_channels"]},
            _stereo,
            1,
        ),
        (
            {"kind": "audio", "channels": 1, "normalize": ["first_channel"]},
            _stereo,
            1,
        ),
    ],
)
def test_model_receives_expected_audio_channels(spec, source, expected, batched):
    """The model sees the declared channel count, or the input's own if none."""
    counts = _channels_seen_by_model(spec, source(), batched)
    assert counts == [expected] * len(counts)


def test_model_receives_channels_kept_by_a_custom_normalizer(monkeypatch):
    """A normalizer that keeps both channels satisfies ``channels: 2``."""
    monkeypatch.setitem(
        NORMALIZERS,
        "scale_audio",
        lambda audio: Audio(audio.array * 0.5, audio.rate),
    )
    spec = {"kind": "audio", "channels": 2, "normalize": ["scale_audio"]}
    assert _channels_seen_by_model(spec, _stereo(), batched=False) == [2]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    ("spec", "source", "message"),
    [
        # A reduction that contradicts the declaration is rejected before the model.
        (
            {"kind": "audio", "channels": 2, "normalize": ["mean_channels"]},
            _stereo,
            "needs 2",
        ),
        # Multichannel audio is never reduced unless a normalizer is configured.
        ({"kind": "audio", "channels": 1}, _stereo, "needs 1"),
    ],
)
def test_model_is_not_called_when_channels_do_not_match(spec, source, message, batched):
    """A final channel mismatch stops the call before the model runs."""
    with pytest.raises(TypeError, match=message):
        _channels_seen_by_model(spec, source(), batched)

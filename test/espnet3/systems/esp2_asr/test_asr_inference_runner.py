"""The ASR runner formats native Speech2Text hypotheses for the contract."""

import pytest

from espnet3.systems.esp2_asr.inference import ASRInferenceRunner


@pytest.mark.parametrize("text", [None, "", "  ", "日本語。"])
def test_nbest_preserves_best_text_and_valid_empty_values(text):
    result = ASRInferenceRunner.format_output(
        [(text, [], [], None), ("second", [], [], None)], batched=False
    )
    assert result == {"text": "" if text is None else text}


@pytest.mark.parametrize("result", [None, {}, "text", [()], [[]], ["hello"]])
def test_malformed_nbest_is_not_a_successful_empty_transcription(result):
    with pytest.raises((TypeError, ValueError)):
        ASRInferenceRunner.format_output(result, batched=False)


@pytest.mark.parametrize("text", [0, False, [], 123])
def test_false_values_other_than_none_are_not_coerced_to_empty_text(text):
    with pytest.raises(TypeError, match="must be str"):
        ASRInferenceRunner.forward(
            0,
            dataset=[{"speech": "input"}],
            model=lambda speech: [(text, [], [], None)],
            input_key="speech",
            inference_contract={
                "inputs": {"speech": {"kind": "text"}},
                "outputs": {"text": {"kind": "text"}},
            },
        )


class FakeSpeech2Text:
    """Return native ESPnet2 hypotheses without building a model."""

    def __call__(self, speech):
        """Return one n-best list for each waveform."""
        if isinstance(speech, list):
            return [[("A", [], [], None)], [("B", [], [], None)]]
        return [("ONE", [], [], None)]


def test_asr_runner_formats_single_native_result():
    """Single-item inference extracts the best hypothesis text."""
    dataset = [{"speech": "audio"}]

    result = ASRInferenceRunner.forward(
        0,
        dataset=dataset,
        model=FakeSpeech2Text(),
        input_key="speech",
    )

    assert result == {"text": "ONE"}


def test_asr_runner_formats_batched_native_results():
    """Batched inference preserves the input order."""
    dataset = [{"speech": "audio-a"}, {"speech": "audio-b"}]

    result = ASRInferenceRunner.forward(
        [0, 1],
        dataset=dataset,
        model=FakeSpeech2Text(),
        input_key="speech",
    )

    assert result == [{"text": "A"}, {"text": "B"}]


@pytest.mark.parametrize("batched", [False, True])
def test_asr_forward_validates_after_formatting(batched):
    """Direct calls normalize inputs and validate the final named ASR results."""
    seen = []

    def model(speech):
        seen.append(speech)
        if isinstance(speech, list):
            return [[("  HELLO!  ", [], [], None)] for _ in speech]
        return [("  HELLO!  ", [], [], None)]

    contract = {
        "inputs": {"speech": {"kind": "text", "normalize": ["strip"]}},
        "outputs": {"text": {"kind": "text", "normalize": ["strip", "to_lower"]}},
    }
    result = ASRInferenceRunner.forward(
        [0, 1] if batched else 0,
        dataset=[{"speech": " A ", "utt_id": "a"}, {"speech": " B ", "utt_id": "b"}],
        model=model,
        input_key="speech",
        inference_contract=contract,
    )
    assert seen == ([["A", "B"]] if batched else ["A"])
    expected = {"text": "hello!", "utt_id": "a"}
    assert result == (
        [expected, {"text": "hello!", "utt_id": "b"}] if batched else expected
    )


def test_asr_forward_rejects_invalid_formatted_output():
    """A malformed hypothesis is rejected after the ASR formatter runs."""
    with pytest.raises(TypeError, match="must be str"):
        ASRInferenceRunner.forward(
            0,
            dataset=[{"speech": "input"}],
            model=lambda speech: [(123, [], [], None)],
            input_key="speech",
            inference_contract={
                "inputs": {"speech": {"kind": "text"}},
                "outputs": {"text": {"kind": "text"}},
            },
        )


def test_asr_runner_forward_survives_pickling():
    """Worker serialization preserves the subclass output formatter."""
    import pickle

    forward = pickle.loads(pickle.dumps(ASRInferenceRunner.forward))
    assert forward(
        0, dataset=[{"speech": "audio"}], model=FakeSpeech2Text(), input_key="speech"
    ) == {"text": "ONE"}


def test_asr_runner_preserves_all_channels_for_native_backend():
    """The ESPnet2 adapter changes array layout without selecting a channel."""
    import numpy as np

    source = np.stack([np.full(160, 0.25), np.full(160, 0.75)])
    seen = []

    def backend(speech):
        seen.append(speech)
        return [("ok", [], [], None)]

    result = ASRInferenceRunner.forward(
        0,
        dataset=[{"speech": (16000, source)}],
        model=backend,
        input_key="speech",
        inference_contract={
            "inputs": {"speech": {"kind": "audio", "sample_rate": 16000}},
            "outputs": {"text": {"kind": "text"}},
        },
    )
    assert seen[0].shape == (160, 2)
    np.testing.assert_array_equal(seen[0], source.T)
    assert result["text"] == "ok"

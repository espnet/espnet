"""The ASR runner formats native Speech2Text hypotheses for the contract."""

from espnet3.systems.esp2_asr.inference import ASRInferenceRunner


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

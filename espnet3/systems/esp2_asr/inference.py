"""ASR-specific output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ASRInferenceRunner(InferenceRunner):
    """Convert ESPnet2 ASR n-best results to the contracted ``text`` field."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass all channels to ESPnet2 in its samples-first array layout."""

        def waveform(value):
            if not isinstance(value, Audio):
                return value
            return value.array.T if value.array.ndim == 2 else value.array

        inputs = dict(inputs)
        speech = inputs["speech"]
        inputs["speech"] = (
            [waveform(value) for value in speech]
            if isinstance(speech, list)
            else waveform(speech)
        )
        return inputs

    @staticmethod
    def _format_one(hypotheses):
        """Return the best hypothesis as a named text field."""
        best = hypotheses[0]
        if isinstance(best, list):
            best = best[0]
        return {"text": best[0] or ""}

    @staticmethod
    def format_output(result, *, batched):
        """Format best text hypotheses before the shared contract checks."""
        if batched:
            return [ASRInferenceRunner._format_one(item) for item in result]
        return ASRInferenceRunner._format_one(result)

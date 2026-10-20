"""ASR-specific output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ASRInferenceRunner(InferenceRunner):
    """Convert ESPnet2 ASR n-best results to the contracted ``text`` field."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass all channels to ESPnet2 in its samples-first array layout.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio`` (a list of them for a batch).

        Returns:
            A copy of ``inputs`` whose ``speech`` is a ``(samples, channels)``
            array for stereo audio, or a 1-D array for mono.

        Examples:
            >>> import numpy as np
            >>> audio = Audio(np.zeros((2, 8000), dtype=np.float32), 16000)
            >>> out = ASRInferenceRunner.prepare_model_inputs({"speech": audio})
            >>> out["speech"].shape
            (8000, 2)
        """

        def waveform(value):
            """Return the samples of an ``Audio``, samples first."""
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
        """Format best text hypotheses before the shared contract checks.

        Args:
            result: The n-best results of ``Speech2Text`` (a list of them for a
                batch).
            batched: Whether the call was a batch.

        Returns:
            ``{"text": ...}`` for one call, or a list of them for a batch.

        Examples:
            >>> result = [("hello", ["h"], [1], None)]
            >>> ASRInferenceRunner.format_output(result, batched=False)
            {'text': 'hello'}
        """
        if batched:
            return [ASRInferenceRunner._format_one(item) for item in result]
        return ASRInferenceRunner._format_one(result)

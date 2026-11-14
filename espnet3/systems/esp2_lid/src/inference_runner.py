"""LID-specific input and output adaptation for the shared inference runner."""

from __future__ import annotations

from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2LIDInferenceRunner(InferenceRunner):
    """Run ``Speech2Language`` and return its prediction as ``language``."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass the checked mono ``Audio`` to ``Speech2Language`` as samples.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio`` (a list of them for a batch).

        Returns:
            A copy of ``inputs`` whose ``speech`` is a 1-D array (a list of
            them for a batch).

        Examples:
            >>> import numpy as np
            >>> from espnet3.contract.kinds.audio import Audio
            >>> audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
            >>> out = ESP2LIDInferenceRunner.prepare_model_inputs({"speech": audio})
            >>> out["speech"].shape
            (16000,)
        """
        inputs = dict(inputs)
        speech = inputs["speech"]
        inputs["speech"] = (
            [item.array for item in speech]
            if isinstance(speech, list)
            else speech.array
        )
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Return the predicted language as the contracted ``language`` field.

        ``Speech2Language`` returns a language code, or a ``hyp``/``embedding``
        dictionary when built with ``extract_embd=True``; only the language is
        an output.

        Args:
            result: What ``Speech2Language`` returned (a list for a batch).
            batched: Whether the call was a batch.

        Returns:
            ``{"language": ...}`` for one call, or a list of them for a batch.

        Examples:
            >>> ESP2LIDInferenceRunner.format_output("jpn", batched=False)
            {'language': 'jpn'}
            >>> ESP2LIDInferenceRunner.format_output(
            ...     [{"hyp": "eng", "embedding": None}], batched=True
            ... )
            [{'language': 'eng'}]
        """

        def one(value):
            """Return the language of one prediction."""
            return {"language": value["hyp"] if isinstance(value, dict) else value}

        return [one(value) for value in result] if batched else one(result)

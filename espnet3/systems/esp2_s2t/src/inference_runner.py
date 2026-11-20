"""SOT output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.esp2_s2t.src.sot_output import render_sot


class ESP2S2TInferenceRunner(InferenceRunner):
    """Convert a native S2T result to the contracted ``hyp`` field."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass ESPnet2 the samples of the checked mono ``Audio``.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio`` the contract has already resampled and checked to
                be one channel.

        Returns:
            A copy of ``inputs`` whose ``speech`` is a 1-D array.

        Examples:
            >>> import numpy as np
            >>> audio = Audio(np.zeros(8000, dtype=np.float32), 16000)
            >>> out = ESP2S2TInferenceRunner.prepare_model_inputs({"speech": audio})
            >>> out["speech"].shape
            (8000,)
        """
        inputs = dict(inputs)
        if isinstance(inputs["speech"], Audio):
            inputs["speech"] = inputs["speech"].array
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Render the best hypothesis, timestamps and separators intact.

        Args:
            result: The n-best list of ``Speech2Text``.
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": ...}``.

        Raises:
            ValueError: If the call was a batch. The timestamp rules track
                one prompt length for the whole beam, so SOT decoding takes
                one utterance group at a time (``batch_size: null``).

        Examples:
            >>> tokens = ["<|en|>", "<|transcribe|>", "<|0.00|>", " hi", "<|0.40|>"]
            >>> result = [("".join(tokens), tokens, [], " hi", None)]
            >>> ESP2S2TInferenceRunner.format_output(result, batched=False)
            {'hyp': '<|0.00|> hi<|0.40|>'}
        """
        if batched:
            raise ValueError(
                "SOT decoding takes one utterance group at a time; "
                "set batch_size: null in the inference config"
            )
        return {"hyp": render_sot(result)}

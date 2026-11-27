"""SOT output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2S2TInferenceRunner(InferenceRunner):
    """Convert a native S2T result to the contracted ``hyp`` field."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass ESPnet2 the samples of the checked mono ``Audio``.

        Args:
            inputs: Keyword arguments for the backend. ``speech`` is an
                ``Audio`` object that the contract has already resampled and
                checked to have one channel.

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

        The text comes from the converter of the decoder. The ``Speech2Text``
        of this system builds that converter so that it keeps special tokens.
        A separator that the checkpoint added to its vocabulary therefore stays
        in the text. Only the prompt is removed, which is the language symbol
        and the task symbol at the start.

        Returns:
            ``{"hyp": ...}``. The value is ``""`` when the decoder returned no
            hypothesis.

        Raises:
            ValueError: If the call was a batch. The timestamp rules track
                one prompt length for the whole beam, so SOT decoding takes
                one utterance group at a time (``batch_size: null``).
            ValueError: If the text does not start with the two prompt
                tokens, which means a converter dropped special tokens.

        Examples:
            >>> tokens = ["<|en|>", "<|transcribe|>", "<|0.00|>", " hi", "<|0.40|>"]
            >>> result = [("".join(tokens), tokens, [], " hi", None)]
            >>> ESP2S2TInferenceRunner.format_output(result, batched=False)
            {'hyp': '<|0.00|> hi<|0.40|>'}
        """
        if batched:
            raise ValueError(
                "SOT decoding takes one utterance group at a time. "
                "Set batch_size: null in the inference config."
            )
        if not result:
            return {"hyp": ""}
        text, tokens, *_ = result[0]
        prompt = "".join(tokens[:2])
        # A converter that drops special tokens leaves words in these two slots.
        special = [t for t in tokens[:2] if t.startswith("<|") and t.endswith("|>")]
        if len(special) != 2 or not text.startswith(prompt):
            raise ValueError(
                f"hypothesis {text!r} does not start with its prompt {prompt!r}. "
                "Decode with espnet3.systems.esp2_s2t.src.sot.s2t_inference."
                "Speech2Text, and set speaker_change_symbol or adaptive_timestamp."
            )
        return {"hyp": text[len(prompt) :]}

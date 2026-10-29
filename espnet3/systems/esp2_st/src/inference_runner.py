"""ST-specific input and output adaptation for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2STInferenceRunner(InferenceRunner):
    """Adapt ``espnet2.bin.st_inference.Speech2Text`` to the contract.

    Two departures from the default runner, both forced by that backend:

    * it is ``@typechecked`` on a single ``speech`` array and has no
      ``batch_decode``, so a list cannot be passed at all;
    * a multi-decoder model returns ``(translations, transcripts)`` rather
      than one n-best list, and only the translation is this system's
      output.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the backend the waveform it takes, rejecting a batch.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                :class:`~espnet3.contract.kinds.audio.Audio`.

        Returns:
            A copy of ``inputs`` whose ``speech`` is the sample array.

        Raises:
            ValueError: If ``speech`` is a list, i.e. the stage was
                configured with a batch size. This runs before
                ``format_output``, so the batch is refused here with a
                message naming the setting rather than inside ESPnet2's
                type check.

        Examples:
            >>> import numpy as np
            >>> audio = Audio(np.zeros(8000, dtype=np.float32), 16000)
            >>> out = ESP2STInferenceRunner.prepare_model_inputs(
            ...     {"speech": audio}
            ... )
            >>> out["speech"].shape
            (8000,)
        """
        speech = inputs["speech"]
        if isinstance(speech, list):
            raise ValueError(
                "ESPnet2 ST decodes one utterance per call: set "
                "`batch_size: null` in the inference config"
            )
        array = speech.array if isinstance(speech, Audio) else speech
        return {**inputs, "speech": array}

    @staticmethod
    def format_output(result, *, batched):
        """Return the best translation as the contracted ``text`` field.

        Args:
            result: What ``Speech2Text`` returned: an n-best list of
                ``(text, token, token_int, hyp)``, or
                ``(translations, transcripts)`` for a multi-decoder model.
            batched: Whether the call was a batch; always false here,
                since :meth:`prepare_model_inputs` refuses one.

        Returns:
            ``{"text": str}``. ``Speech2Text`` leaves the text ``None``
            when it has no tokenizer, which is an empty string here.

        Examples:
            >>> result = [("hallo welt", ["hallo"], [1], None)]
            >>> ESP2STInferenceRunner.format_output(result, batched=False)
            {'text': 'hallo welt'}
        """
        translations = result[0] if isinstance(result, tuple) else result
        return {"text": translations[0][0] or ""}

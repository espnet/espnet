"""S2T output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2S2TInferenceRunner(InferenceRunner):
    """Turn an OWSM-style backend's utterances into ``text`` and ``segments``.

    The backend (:class:`~espnet3.systems.esp2_s2t.src.longform_backend.
    S2TLongFormBackend`) returns ``(start, end, text)`` for every utterance it
    found in a recording; both contracted outputs are read off that list, so a
    long recording is one row of joined text and a segment per utterance.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the backend plain waveforms rather than :class:`Audio`.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio`` (a list of them for a batch).

        Returns:
            A copy of ``inputs`` whose ``speech`` is the samples array.

        Examples:
            >>> import numpy as np
            >>> audio = Audio(np.zeros(8000, dtype=np.float32), 16000)
            >>> out = ESP2S2TInferenceRunner.prepare_model_inputs(
            ...     {"speech": audio}
            ... )
            >>> out["speech"].shape
            (8000,)
        """

        def waveform(value):
            """Return the samples of an ``Audio``, mono as 1-D."""
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
    def _format_one(utterances):
        """Return the joined text and one segment per utterance.

        Args:
            utterances: ``(start, end, text)`` tuples, in time order. Empty
                for a recording the model found nothing in, including one
                too short for the encoder to subsample -- that is scored as
                an empty hypothesis rather than dropped, so the reference
                still counts against the model.

        Returns:
            ``{"text": ..., "segments": [...]}``.
        """
        if utterances is None:
            return {"text": "", "segments": []}
        if not isinstance(utterances, (list, tuple)):
            raise TypeError("s2t results must be a list of (start, end, text)")
        segments = []
        for utterance in utterances:
            if not isinstance(utterance, (list, tuple)) or len(utterance) != 3:
                raise TypeError("each s2t utterance must be (start, end, text)")
            start, end, text = utterance
            segments.append({"text": text, "start": start, "end": end})
        return {
            "text": " ".join(segment["text"] for segment in segments),
            "segments": segments,
        }

    @staticmethod
    def format_output(result, *, batched):
        """Format one recording's utterances, or a batch of them.

        Args:
            result: The backend's return: ``(start, end, text)`` tuples, or a
                list of those lists when ``batched``.
            batched: Whether ``result`` covers several recordings.

        Returns:
            One ``{"text", "segments"}`` mapping, or a list of them.
        """
        if not batched:
            return ESP2S2TInferenceRunner._format_one(result)
        if not isinstance(result, (list, tuple)):
            raise TypeError("a batched s2t result must be a list")
        return [ESP2S2TInferenceRunner._format_one(one) for one in result]

"""The kNN-VC inference runner: checked inputs in, converted waveform out."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


def _samples(value):
    """Return the samples of an ``Audio``; anything else (a bare array) as is."""
    return value.array if isinstance(value, Audio) else value


class KNNVCInferenceRunner(InferenceRunner):
    """Adapt the contract's values to ``KNNVCModel`` and name its result ``wav``.

    ``forward`` is inherited, so the shared checks stay in place: the
    contract of ``inference.yaml`` checks ``speech`` (``audio``),
    ``reference_speech`` (``reference_audio``, a list of utterances) and
    ``target_speaker`` (``text``) on the way in, and the ``wav`` output on
    the way out, attaching ``pair_id`` as the sample identifier.

    Examples:
        >>> runner_cls = KNNVCInferenceRunner
        >>> runner_cls.prepare_model_inputs({"speech": [audio_a, audio_b]})
        Traceback (most recent call last):
        ValueError: kNN-VC converts one pair per call ...
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand ``KNNVCModel`` bare waveforms and keep the reference boundaries.

        Args:
            inputs: The checked model arguments: ``speech`` as an ``Audio``,
                ``reference_speech`` as a list of them, ``target_speaker`` as
                a string (a bare array per utterance when no contract ran).

        Returns:
            A copy with ``speech`` as a 1-D array and ``reference_speech`` as
            a list of 1-D arrays, one per reference utterance, so the model
            trims and encodes each on its own.

        Raises:
            ValueError: If the call is a batch. Reference sets differ in
                size, so a conversion runs one pair per call
                (``batch_size: null``).
        """
        if isinstance(inputs.get("speech"), list):
            raise ValueError(
                "kNN-VC converts one pair per call, as reference sets differ in "
                "size; set `batch_size: null` in the inference config"
            )
        prepared = dict(inputs)
        prepared["speech"] = _samples(inputs["speech"])
        prepared["reference_speech"] = [
            _samples(ref) for ref in inputs["reference_speech"]
        ]
        return prepared

    @staticmethod
    def format_output(result, *, batched):
        """Name the converted waveform ``wav``.

        ``KNNVCModel`` returns a bare float32 array at its encoder's rate;
        the contract's ``wav`` output (``sample_rate: 16000``) turns it into
        an ``Audio`` and checks it.

        Args:
            result: The converted waveform.
            batched: Always ``False``: :meth:`prepare_model_inputs` refuses
                batches before the model runs.

        Returns:
            ``{"wav": result}``.
        """
        return {"wav": result}

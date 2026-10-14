"""The ESPnet2 enhancement system's side of :mod:`espnet3.api.inference`.

The model behind an ``enh`` recipe is
:class:`espnet2.bin.enh_inference.SeparateSpeech`, which takes a batch of
noisy waveforms and returns one batch of waveforms per output speaker. This
adapter returns the first speaker as ``enhanced``. Building the
``SeparateSpeech`` and loading a bundle come from
:class:`espnet3.systems.base.backend_inference.BackendInference`.

Examples:
    In the ``infer`` stage, ``inference.yaml`` names this class with
    ``SeparateSpeech``'s own arguments::

        model:
          _target_: espnet3.systems.esp2_enh.inference.Inference
          train_config: ${exp_dir}/config.yaml
          model_file: ${exp_dir}/last.ckpt
          normalize_output_wav: true

    From a ``pack_model`` bundle::

        >>> model = Inference.from_pretrained("exp/training/model_pack")
        >>> model("noisy.wav")["enhanced"].rate
        16000
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference


class Inference(BackendInference):
    """Enhance speech with an ESPnet2 enhancement model.

    Args:
        backend: An already-built ``SeparateSpeech``, or ``None`` to build one
            from ``kwargs``.
        sample_rate: The rate the model was trained at. Input audio is
            resampled to it and the enhanced audio is returned at it.
        **kwargs: ``device``, ``backend_class`` and ``SeparateSpeech``'s own
            arguments, as :class:`BackendInference` takes them.

    Examples:
        >>> model = Inference(train_config="exp/config.yaml",
        ...                   model_file="exp/last.ckpt")
        >>> model.batch([{"speech_mix": "a.wav"}, {"speech_mix": "b.wav"}])
        [{'enhanced': Audio(...)}, {'enhanced': Audio(...)}]
    """

    backend_class = "espnet2.bin.enh_inference.SeparateSpeech"
    inputs = (Field("speech_mix", "audio", "Noisy speech"),)
    outputs = (Field("enhanced", "audio", "Enhanced speech"),)

    def __init__(
        self, backend: Any = None, *, sample_rate: int = 16000, **kwargs: Any
    ) -> None:
        """Build or keep the backend and remember the model's rate."""
        super().__init__(backend, **kwargs)
        self._sample_rate = int(sample_rate)

    @property
    def sample_rate(self) -> int:
        """The rate the model was trained at."""
        return self._sample_rate

    def run(self, speech_mix: Audio) -> Mapping[str, Any]:
        """Return the enhanced version of one utterance."""
        return self.run_batch([{"speech_mix": speech_mix}])[0]

    def run_batch(self, items: Sequence[Mapping[str, Any]]) -> Sequence[Mapping]:
        """Enhance a batch in one model call.

        Shorter utterances are zero-padded for the call, and each output is
        cut back to its own input's length.
        """
        waves = [item["speech_mix"].mono().array for item in items]
        lengths = [len(wave) for wave in waves]
        batch = np.zeros((len(waves), max(lengths)), dtype=np.float32)
        for i, wave in enumerate(waves):
            batch[i, : len(wave)] = wave
        first_speaker = np.asarray(self.backend(batch, fs=self.sample_rate)[0])
        return [
            {"enhanced": Audio(_limit_peak(first_speaker[i, :n]), self.sample_rate)}
            for i, n in enumerate(lengths)
        ]


def _limit_peak(wave: np.ndarray) -> np.ndarray:
    """Scale a waveform down to a 0.9 peak if it would clip, else keep it."""
    wave = np.asarray(wave, dtype=np.float32)
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 1.0:
        wave = wave / peak * 0.9
    return wave

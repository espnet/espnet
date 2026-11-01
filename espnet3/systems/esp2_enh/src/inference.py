"""Speech enhancement with an ESPnet2 ``SeparateSpeech`` model.

The model behind an ``enh`` recipe is
:class:`espnet2.bin.enh_inference.SeparateSpeech`, which takes a batch of
noisy waveforms and returns one batch of waveforms per output speaker. This
class returns the first speaker as ``enhanced``. The shared
:class:`~espnet3.systems.base.inference_runner.InferenceRunner` calls it with
``speech_mix`` already checked by the contract in ``conf/inference.yaml``.

Examples:
    In the ``infer`` stage, ``inference.yaml`` names this class with
    ``SeparateSpeech``'s own arguments (the provider adds ``device``)::

        model:
          _target_: espnet3.systems.esp2_enh.src.inference.Inference
          train_config: ${exp_dir}/config.yaml
          model_file: ${exp_dir}/last.ckpt
          normalize_output_wav: true
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from espnet2.bin.enh_inference import SeparateSpeech
from espnet3.contract.kinds.audio import Audio


class Inference:
    """Enhance speech with an ESPnet2 enhancement model.

    Args:
        backend: An already-built ``SeparateSpeech``, or ``None`` to build one
            from ``kwargs``.
        sample_rate: The rate the model was trained at. The enhanced audio is
            returned at it.
        **kwargs: ``SeparateSpeech``'s own arguments, ``device`` included.

    Examples:
        >>> model = Inference(train_config="exp/config.yaml",
        ...                   model_file="exp/last.ckpt", device="cpu")
        >>> model(speech_mix=[Audio(a, 16000), Audio(b, 16000)])
        [{'enhanced': Audio(...)}, {'enhanced': Audio(...)}]
    """

    def __init__(
        self, backend: Any = None, *, sample_rate: int = 16000, **kwargs: Any
    ) -> None:
        """Build or keep the backend and remember the model's rate."""
        self.backend = SeparateSpeech(**kwargs) if backend is None else backend
        self.sample_rate = int(sample_rate)

    def __call__(self, speech_mix):
        """Enhance one ``Audio``, or a list of them as one batch."""
        if isinstance(speech_mix, list):
            return self.run_batch([{"speech_mix": item} for item in speech_mix])
        return self.run(speech_mix)

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

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import soundfile as sf


class TSEAudioLoadTransform:
    """Decode the audio fields of a LibriMix TSE sample into waveform arrays.

    It participates in the ``collect_stats`` and ``train`` stages, wired from
    ``training.yaml``'s ``dataset.train[i].transform`` / ``dataset.valid[i].transform``
    blocks, and runs before ``espnet2.train.preprocessor.TSEPreprocessor``.

    Note that leaving ``load_enrollment_as_audio`` False keeps enrollment decoding
    inside ``TSEPreprocessor``, which is the only path that applies ``enroll_segment``
    cropping. Set it to True only when the preprocessor on that split accepts
    array-valued enrollment.

    Args:
        load_enrollment_as_audio: Whether to decode ``enroll_ref{N}`` path values.
            Defaults to False, which leaves them for the preprocessor.
        speech_key_prefix: Field-name prefixes treated as mixture/reference
            speech and always decoded. Defaults to
            ``("speech_mix", "speech_ref")``.
        enrollment_key_prefix: Field-name prefix treated as enrollment audio.
            Defaults to ``"enroll_ref"``.
        dtype: ``soundfile`` output dtype. Defaults to ``"float32"``, matching
            what ``LibriMixTSEDataset`` produces.

    Returns:
        A new dict with the same keys as the input, where the selected audio
        fields hold 1-D ``numpy`` waveform arrays.

    Raises:
        TypeError: If a field selected for decoding is neither a path-like
            value nor a ``numpy.ndarray``, which means the dataset produced a
            field shape this transform does not know how to handle.
        ValueError: If the fields decoded from one sample do not share a single
            sampling rate, which would otherwise surface much later as a shape
            mismatch inside the model.

    Examples:
        Wired from ``conf/tuning/training_td_speakerbeam_16k.yaml``::

            dataset:
              train:
                - data_src_args:
                    split: 2mix_16k_max_train_mix-clean
                    ignore_key_prefix: ["text_spk", "utt_id", "num_spk"]
                  transform:
                    _target_: src.transforms.TSEAudioLoadTransform
                    load_enrollment_as_audio: true

        Called directly:

        >>> transform = TSEAudioLoadTransform(load_enrollment_as_audio=True)
        >>> sample = transform(
        ...     {
        ...         "speech_mix": "/data/2mix/mix_clean/utt.wav",
        ...         "speech_ref1": "/data/2mix/s1/utt.wav",
        ...         "enroll_ref1": "*1272-128104-0000 1272",
        ...     }
        ... )
        >>> type(sample["speech_mix"]).__name__
        'ndarray'
        >>> sample["enroll_ref1"]  # placeholder left for TSEPreprocessor
        '*1272-128104-0000 1272'
    """

    def __init__(
        self,
        load_enrollment_as_audio: bool = False,
        speech_key_prefix: Sequence[str] = ("speech_mix", "speech_ref"),
        enrollment_key_prefix: str = "enroll_ref",
        dtype: str = "float32",
    ) -> None:
        """Initialize the transform."""
        self.load_enrollment_as_audio = bool(load_enrollment_as_audio)
        self.speech_key_prefix = tuple(speech_key_prefix)
        self.enrollment_key_prefix = str(enrollment_key_prefix)
        self.dtype = str(dtype)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the audio fields of one sample.

        Args:
            sample: One item from ``LibriMixTSEDataset.__getitem__``.

        Returns:
            A new dict with the same keys as ``sample``, where the selected
            audio fields hold 1-D ``numpy`` waveform arrays.
        """
        decoded: Dict[str, Any] = {}
        sample_rates: list[int] = []

        for key, value in sample.items():
            if key.startswith(self.enrollment_key_prefix):
                if not self.load_enrollment_as_audio or self._is_placeholder(value):
                    decoded[key] = value
                    continue
            elif not key.startswith(self.speech_key_prefix):
                decoded[key] = value
                continue

            audio, sample_rate = self._load_audio(key, value)
            decoded[key] = audio
            if sample_rate is not None:
                sample_rates.append(sample_rate)

        if len(set(sample_rates)) > 1:
            raise ValueError(
                "All audio fields of one sample must share a sampling rate, "
                f"but got {sorted(set(sample_rates))}. Check that the "
                "configured split mixes only one sampling rate."
            )
        return decoded

    @staticmethod
    def _is_placeholder(value: Any) -> bool:
        """Check whether a value is a placeholder for enrollment audio."""
        # "*UTT_ID SPEAKER_ID" marks a training enrollment that TSEPreprocessor
        # resolves against its spk2enroll map; this transform cannot resolve it.
        return isinstance(value, str) and value.startswith("*")

    def _load_audio(self, key: str, value: Any) -> Tuple[np.ndarray, int | None]:
        """Load audio from a path or return it if already a numpy array."""
        # Returns (waveform, sampling_rate); sampling_rate is None when the
        # value was already decoded upstream and carries no rate to check.
        if isinstance(value, np.ndarray):
            return value, None
        if isinstance(value, (str, Path)):
            audio, sample_rate = sf.read(str(value), dtype=self.dtype)
            return audio, int(sample_rate)
        raise TypeError(
            f"Field '{key}' must be a path or numpy.ndarray to be decoded, "
            f"but got {type(value).__name__}."
        )

    def __repr__(self) -> str:
        """Return a compact representation for the dataset logs."""
        return (
            f"{type(self).__name__}("
            f"load_enrollment_as_audio={self.load_enrollment_as_audio}, "
            f"dtype={self.dtype!r})"
        )

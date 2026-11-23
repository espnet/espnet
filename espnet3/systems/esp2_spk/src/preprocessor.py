"""Speaker preprocessor adjustments for the ESPnet3 data pipeline."""

from typing import Optional, Sequence

import numpy as np

from espnet2.train.preprocessor import SpkPreprocessor as _SpkPreprocessor

_WAVEFORM_KEYS = ("speech", "speech2")


def speed_perturbed_speaker(speaker: str, factor: float) -> str:
    """Return the speaker label of ``speaker`` speed-perturbed by ``factor``.

    Follows ESPnet2's ``spk.sh``, where each perturbed copy of a speaker is a
    speaker of its own (``sp0.9-id10001``) and ``1.0`` keeps the original name.

    Args:
        speaker: Original speaker label.
        factor: Speed factor.

    Returns:
        The label of the perturbed speaker.

    Examples:
        >>> speed_perturbed_speaker("id10001", 0.9)
        'sp0.9-id10001'
        >>> speed_perturbed_speaker("id10001", 1.0)
        'id10001'
    """
    factor = float(factor)
    return speaker if factor == 1.0 else f"sp{factor}-{speaker}"


class SpkPreprocessor(_SpkPreprocessor):
    """Speaker preprocessor that always emits float32 waveforms.

    ESPnet2 routes preprocessor output through ``ESPnetDataset``, which casts
    float64 arrays down to float32 before batching. ESPnet3 hands the output
    straight to the collate function instead, and the noise and reverberation
    augmentation of the base class promotes waveforms to float64. Since the
    augmentations are applied probabilistically, that would otherwise leave the
    dtype of a minibatch dependent on which samples happened to be augmented.

    Args:
        speed_perturb_factors: Speed factors the training dataset perturbs
            with, if any. ESPnet2 writes the perturbed speakers into the
            ``spk2utt`` of a ``<train_set>_sp`` directory; here ``spk2utt``
            stays the original one and the label space is extended instead,
            with a speaker per factor named by :func:`speed_perturbed_speaker`.
            The original speakers keep labels ``0 .. N-1``.
        **kwargs: Passed to :class:`espnet2.train.preprocessor.SpkPreprocessor`.

    Examples:
        >>> preprocessor = SpkPreprocessor(train=True, target_duration=3.0)
        >>> preprocessor("utt1", {"speech": wav, "spk_labels": "id10001"})
    """

    def __init__(
        self,
        *args,
        speed_perturb_factors: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Build the base preprocessor, then extend its speaker labels."""
        super().__init__(*args, **kwargs)
        self.speed_perturb_factors = (
            None
            if speed_perturb_factors is None
            else [float(f) for f in speed_perturb_factors]
        )
        if self.spk2label is not None and self.speed_perturb_factors:
            base = list(self.spk2label)
            for factor in self.speed_perturb_factors:
                if factor == 1.0:
                    continue
                for speaker in base:
                    self.spk2label[speed_perturbed_speaker(speaker, factor)] = len(
                        self.spk2label
                    )
            self.nspk = len(self.spk2label)

    def __call__(self, uid, data):
        """Preprocess one sample and normalize its waveform dtype.

        Args:
            uid: Sample identifier, unused by the base class.
            data: Sample holding the waveform(s) and the speaker label.

        Returns:
            The preprocessed sample, with every waveform as float32.
        """
        data = super().__call__(uid, data)
        for key in _WAVEFORM_KEYS:
            array = data.get(key)
            if array is not None and array.dtype != np.float32:
                data[key] = array.astype(np.float32)
        return data

"""Speaker preprocessor adjustments for the ESPnet3 data pipeline."""

import numpy as np

from espnet2.train.preprocessor import SpkPreprocessor as _SpkPreprocessor

_WAVEFORM_KEYS = ("speech", "speech2")


class SpkPreprocessor(_SpkPreprocessor):
    """Speaker preprocessor that always emits float32 waveforms.

    ESPnet2 routes preprocessor output through ``ESPnetDataset``, which casts
    float64 arrays down to float32 before batching. ESPnet3 hands the output
    straight to the collate function instead, and the noise and reverberation
    augmentation of the base class promotes waveforms to float64. Since the
    augmentations are applied probabilistically, that would otherwise leave the
    dtype of a minibatch dependent on which samples happened to be augmented.

    Examples:
        >>> preprocessor = SpkPreprocessor(train=True, target_duration=3.0)
        >>> preprocessor("utt1", {"speech": wav, "spk_labels": "id10001"})
    """

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

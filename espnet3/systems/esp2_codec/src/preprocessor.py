"""Training preprocessor that feeds the ``speech`` field to an ESPnet2 codec."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from espnet2.train.preprocessor import CommonPreprocessor


class CodecPreprocessor(CommonPreprocessor):
    """``CommonPreprocessor`` that renames the waveform for the codec model.

    ESPnet3 codec datasets declare their waveform as ``speech``, the field
    name the other systems use. The ESPnet2 codec stack takes it as ``audio``:
    ``ESPnetGANCodecModel.forward(audio, ...)``, ``collect_feats(audio, ...)``
    and ``GANCodecTask.required_data_names()``. The batch reaches the model as
    ``model(**batch)``, so the key has to be ``audio`` by the time the collate
    function sees it. This preprocessor applies the usual ``CommonPreprocessor``
    processing to ``speech_name`` and then stores the result under
    ``model_input_name``. ``DataOrganizer`` runs it on every training,
    validation and ``collect_stats`` item.

    Args:
        train: Whether this is the training split's preprocessor;
            ``DataOrganizer`` sets it.
        model_input_name: The keyword the codec model takes the waveform
            under. Defaults to ``audio``.
        **kwargs: Forwarded to ``CommonPreprocessor`` (``speech_name``,
            ``force_single_channel``, ``audio_pad_value``, ...).

    Examples:
        Declared in ``egs3/TEMPLATE/esp2_codec/conf/training.yaml``:

        .. code-block:: yaml

            dataset:
              preprocessor:
                _target_: espnet3.systems.esp2_codec.src.preprocessor.CodecPreprocessor
                token_type: null
                speech_name: speech
                force_single_channel: true
                audio_pad_value: 0.0

        >>> import numpy as np
        >>> preprocessor = CodecPreprocessor(train=False, token_type=None)
        >>> out = preprocessor("utt1", {"speech": np.zeros(4, dtype=np.float32)})
        >>> sorted(out)
        ['audio']
    """

    def __init__(self, train: bool, model_input_name: str = "audio", **kwargs):
        """Build the ``CommonPreprocessor`` and keep the model's input name."""
        super().__init__(train=train, **kwargs)
        self.model_input_name = model_input_name

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Process ``speech_name`` as usual and hand it over as ``model_input_name``.

        Args:
            uid: The item's utterance id.
            data: The item; the waveform sits under ``speech_name``.

        Returns:
            The processed item with the waveform under ``model_input_name``
            and no ``speech_name`` key.

        Raises:
            KeyError: If the item already carries ``model_input_name`` next to
                ``speech_name``, since one would silently overwrite the other.
        """
        data = super().__call__(uid, data)
        if self.speech_name == self.model_input_name or self.speech_name not in data:
            return data
        if self.model_input_name in data:
            raise KeyError(
                f"item {uid!r} carries both {self.speech_name!r} and "
                f"{self.model_input_name!r}; the codec model takes one waveform"
            )
        data = dict(data)
        data[self.model_input_name] = data.pop(self.speech_name)
        return data

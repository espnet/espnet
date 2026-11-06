"""Recipe-local contract kinds shared by the kNN-VC recipes.

Two kinds that the built-in ``audio`` / ``text`` / ``segments`` / ``number``
do not cover:

- ``reference_audio``: the target speaker's reference utterances, a non-empty
  list of waveforms. kNN-VC trims and encodes each one on its own, so they
  are a list of ``audio`` values rather than one concatenated signal.
- ``features``: a finite real-valued ``(frames, dim)`` matrix, what the
  vocoder training view reads back from the ``prepare_features`` output.

Import this module by its absolute name, ``egs3.TEMPLATE.knnvc.src.contracts``,
before constructing a ``Field`` of either kind: the recipe datasets do so at
import time, and the inference config loads it through ``contract.imports``,
which is why a published kNN-VC bundle needs ``trust_user_code=True``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from espnet3.contract import KINDS, Kind, register_kind
from espnet3.contract.kinds.audio import AudioKind


def _is_rate_pair(value) -> bool:
    """Tell a ``(rate, samples)`` pair, which is one utterance, from a list."""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], (int, np.integer))
    )


class ReferenceAudioKind(Kind):
    """``reference_audio``: a non-empty list of reference utterances.

    Every element goes through the ``audio`` checks of its field (finite
    samples, resampling to the field's ``sample_rate`` or the model's, the
    field's ``channels``), so a hook receives a list with one
    :class:`~espnet3.contract.kinds.audio.Audio` per utterance. A
    ``(rate, samples)`` pair is one utterance, not a list of two, and a list
    of such lists is a batch.

    Examples:
        >>> from espnet3.contract import Field
        >>> field = Field("reference_speech", "reference_audio", channels=1)
        >>> refs = KINDS["reference_audio"].check_configured(
        ...     [np.zeros(16000, dtype=np.float32)], field, None,
        ...     {"sample_rate": 16000}, output=False,
        ... )
        >>> [ref.rate for ref in refs]
        [16000]
    """

    option_names = AudioKind.option_names
    # The audio normalizers, applied to each utterance of the list.
    normalizers = {
        name: (lambda one: (lambda refs: [one(ref) for ref in refs]))(fn)
        for name, fn in AudioKind.normalizers.items()
    }
    _audio = AudioKind()

    def validate_options(self, options) -> None:
        """Accept exactly the options an ``audio`` field takes."""
        self._audio.validate_options(options)

    @staticmethod
    def _require_list(value, field, output) -> None:
        if not isinstance(value, (list, tuple)) or _is_rate_pair(value):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, must be a "
                "list of reference utterances"
            )
        if not value:
            raise TypeError(f"{field.name!r} needs at least one reference utterance")

    def check_configured(self, value, field, model, options, *, output):
        """Check each utterance as the ``audio`` field would, with its rate policy."""
        self._require_list(value, field, output)
        element = replace(field, kind="audio")
        return [
            self._audio.check_configured(item, element, model, options, output=output)
            for item in value
        ]

    def check(self, value, field, model, *, output):
        """Check each utterance as the ``audio`` field would, at the model's rate."""
        self._require_list(value, field, output)
        element = replace(field, kind="audio")
        return [
            self._audio.check(item, element, model, output=output) for item in value
        ]

    def is_batch(self, value, field, model) -> bool:
        """A list of reference lists is a batch; a list of utterances is one value."""
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(item, (list, tuple)) and not _is_rate_pair(item)
                for item in value
            )
        )

    def accepts(self, value, field) -> bool:
        """Accept a non-empty list whose every element could be audio."""
        if not isinstance(value, (list, tuple)) or not value or _is_rate_pair(value):
            return False
        element = replace(field, kind="audio")
        return all(self._audio.accepts(item, element) for item in value)


class FeaturesKind(Kind):
    """``features``: a finite real-valued ``(frames, dim)`` matrix; does not join.

    Examples:
        >>> from espnet3.contract import Field
        >>> field = Field("feats", "features")
        >>> KINDS["features"].check(
        ...     np.zeros((10, 1024), dtype=np.float32), field, None, output=False
        ... ).shape
        (10, 1024)
    """

    def check(self, value, field, model, *, output):
        """Require a 2-D floating array (or tensor) without NaN or Inf."""
        if hasattr(value, "detach"):  # a torch tensor, without importing torch
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        where = "returned" if output else "given"
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"{field.name!r} {where} as {array.dtype}, must be a real-valued array"
            )
        if array.ndim != 2:
            raise TypeError(
                f"{field.name!r} {where} with shape {array.shape}, "
                "must be (frames, dim)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{field.name!r} {where} with NaN or Inf values")
        return array

    def accepts(self, value, field) -> bool:
        """Accept what :meth:`check` would; a non-finite matrix is refused too."""
        try:
            self.check(value, field, None, output=False)
        except (TypeError, ValueError):
            return False
        return True


def _register(name: str, kind: Kind) -> None:
    """Register ``kind`` once; a second import of this module is a no-op."""
    existing = KINDS.get(name)
    if existing is not None and type(existing).__name__ != type(kind).__name__:
        raise ValueError(
            f"kind {name!r} is already registered as {type(existing).__name__}"
        )
    register_kind(name, kind, replace=existing is not None)


_register("reference_audio", ReferenceAudioKind())
_register("features", FeaturesKind())

__all__ = ["FeaturesKind", "ReferenceAudioKind"]

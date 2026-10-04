"""The ``audio`` kind: a mono waveform at a rate, from whatever a caller holds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from espnet3.api.inference.kinds.base import BaseKind


@dataclass(frozen=True)
class Audio:
    """A mono float32 waveform and the rate it is sampled at.

    The one shape audio has once it is inside the API. Anything a caller is
    likely to hold - a path, the ``(rate, samples)`` pair ``gr.Audio``
    returns, a NumPy array, a torch tensor - becomes one of these through
    :meth:`coerce`, resampled to the rate the model wants, so a hook only
    ever sees ``audio.array`` at ``audio.rate``.

    Args:
        array: The samples. Integer PCM is scaled to ``[-1, 1]``; from a
            2-D array the first channel is kept, taking the shorter axis as
            channels (``(samples, channels)`` from soundfile and Gradio,
            ``(channels, samples)`` from torchaudio). The first channel,
            not the mean: a microphone array's channels differ in delay,
            and their mean cancels what a single channel keeps. A model
            that wants every channel takes a multichannel kind, not audio.
        rate: Samples per second.

    Raises:
        ValueError: If the array is not 1-D or 2-D, or the rate is not
            positive.

    Examples:
        >>> Audio(np.zeros(16000, dtype=np.float32), 16000).duration
        1.0
        >>> Audio(np.zeros((2, 8000), dtype=np.int16), 8000).array.shape
        (8000,)
        >>> Audio.read("utt.wav", rate=16000).rate
        16000
    """

    array: np.ndarray
    rate: int

    def __post_init__(self) -> None:
        """Bring the samples to mono float32 and check the shape and rate."""
        array = np.asarray(self.array)
        if np.issubdtype(array.dtype, np.integer):
            # PCM as the file or the microphone delivered it
            array = array / np.iinfo(array.dtype).max
        array = array.astype(np.float32, copy=False)
        if array.ndim == 2:
            # (samples, channels) as soundfile and Gradio lay it out,
            # (channels, samples) as torchaudio does: channels are the short
            # axis, there being far fewer of them than samples. The first
            # channel is the reference, as in ESPnet's enhancement; the mean
            # of an array's channels would cancel across their delays.
            array = array[0] if array.shape[0] < array.shape[1] else array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {array.shape}")
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "rate", int(self.rate))
        if self.rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.rate}")

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.array) / self.rate

    @classmethod
    def read(cls, path: str | Path, rate: int | None = None) -> "Audio":
        """Read an audio file, at its own rate or resampled to ``rate``.

        Args:
            path: Any file ``soundfile`` reads (WAV, FLAC, OGG, ...).
            rate: When given, the result is resampled to this rate.

        Returns:
            The file as mono float32.

        Examples:
            >>> Audio.read("utt.flac").rate       # whatever the file holds
            44100
            >>> Audio.read("utt.flac", 16000).rate
            16000
        """
        import soundfile

        array, file_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        audio = cls(array, file_rate)
        return audio if rate is None else audio.resample(rate)

    def resample(self, rate: int) -> "Audio":
        """Return this audio at ``rate``; itself when already there.

        Args:
            rate: The target rate in samples per second.

        Returns:
            A new :class:`Audio` at ``rate``, or this one when it already is.

        Examples:
            >>> Audio(np.zeros(16000, dtype=np.float32), 16000).resample(8000).duration
            1.0
            >>> audio.resample(audio.rate) is audio
            True
        """
        if rate == self.rate:
            return self
        import librosa

        return Audio(
            librosa.resample(self.array, orig_sr=self.rate, target_sr=rate), rate
        )

    @classmethod
    def concat(cls, pieces: Sequence["Audio"]) -> "Audio":
        """Join consecutive pieces of one signal; they must share a rate.

        Raises:
            ValueError: If the pieces are at different rates.
        """
        rates = {p.rate for p in pieces}
        if len(rates) != 1:
            raise ValueError(f"cannot concatenate audio at rates {sorted(rates)}")
        return cls(np.concatenate([p.array for p in pieces]), rates.pop())

    @classmethod
    def coerce(cls, value: Any, rate: Optional[int]) -> "Audio":
        """Turn whatever a caller holds into an :class:`Audio` at ``rate``.

        Args:
            value: One of: an :class:`Audio`; a path, which is read; a
                ``(rate, samples)`` pair, which is what ``gr.Audio`` returns;
                a NumPy array or a torch tensor, taken to be at ``rate``
                already, because nothing says otherwise.
            rate: The rate the result is at; ``None`` keeps each value at
                its own rate, for a model that takes any, and then an
                array or tensor - which carries none - is refused.

        Returns:
            The audio, resampled when its own rate differs.

        Raises:
            TypeError: If ``value`` is none of those, or carries no rate
                when none is fixed.

        Examples:
            >>> Audio.coerce("utt.wav", 16000).rate
            16000
            >>> Audio.coerce((44100, samples), 16000).rate    # from gr.Audio
            16000
            >>> Audio.coerce(torch.zeros(1, 16000), 16000).duration
            1.0
        """
        if isinstance(value, Audio):
            return value if rate is None else value.resample(rate)
        if isinstance(value, (str, Path)):
            return cls.read(value, rate)
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and isinstance(value[0], (int, np.integer))
        ):
            audio = cls(np.asarray(value[1]), int(value[0]))
            return audio if rate is None else audio.resample(rate)
        if hasattr(value, "detach"):  # a torch tensor, without importing torch
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if rate is None:
                raise TypeError(
                    "a bare array carries no sample rate, and this model fixes "
                    "none: give a (rate, samples) pair, an Audio or a path"
                )
            return cls(value, rate)
        raise TypeError(
            "audio must be a path, an Audio, a (rate, samples) pair, "
            f"an array or a tensor, not {type(value).__name__}"
        )


class AudioKind(BaseKind):
    """``audio``: an :class:`Audio` at the model's rate; pieces concatenate."""

    def check(self, value, field, model, *, output):
        """Coerce what a caller holds; a hook's bare array is at the model's rate."""
        rate = model.sample_rate
        if output and isinstance(value, np.ndarray):
            if rate is None:
                raise TypeError(
                    f"{field.name!r} returned as a bare array by a model that "
                    "fixes no sample_rate; return an Audio with its rate"
                )
            return Audio(value, rate)
        return Audio.coerce(value, rate)

    def join(self, first, second):
        """Concatenate the samples."""
        return Audio.concat([first, second])

"""The ``audio`` kind: a waveform at a rate, mono or multichannel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from espnet3.components.contract.kinds.base import Kind


@dataclass(frozen=True)
class Audio:
    """A float32 waveform and the rate it is sampled at.

    The one shape audio has once it is inside the API. Anything a caller is
    likely to hold - a path, the ``(rate, samples)`` pair ``gr.Audio``
    returns, a NumPy array, a torch tensor - becomes one of these through
    :meth:`coerce`, resampled to the rate the model wants, so a hook only
    ever sees ``audio.array`` at ``audio.rate``.

    Every channel is kept: ``array`` is ``(samples,)`` for mono and
    ``(channels, samples)`` otherwise. Which of the two a hook receives is
    the field's choice (``Field(..., channels=...)``): the default, one
    channel, hands a single-channel backend the 1-D array it takes;
    ``None`` or a count hands a multichannel model every channel.

    Args:
        array: The samples. Integer PCM is scaled to ``[-1, 1]``; a 2-D
            array is stored channels-first, taking the shorter axis as
            channels (``(samples, channels)`` from soundfile and Gradio,
            ``(channels, samples)`` from torchaudio), and one channel is
            mono.
        rate: Samples per second.

    Raises:
        ValueError: If the array is not 1-D or 2-D, or the rate is not
            positive.

    Examples:
        >>> Audio(np.zeros(16000, dtype=np.float32), 16000).duration
        1.0
        >>> stereo = Audio(np.zeros((8000, 2), dtype=np.int16), 8000)
        >>> stereo.array.shape, stereo.channels
        ((2, 8000), 2)
        >>> stereo.mono().array.shape           # the reference channel
        (8000,)
        >>> Audio.read("utt.wav", rate=16000).rate
        16000
    """

    array: np.ndarray
    rate: int

    def __post_init__(self) -> None:
        """Bring the samples to float32, channels first, and check shape and rate."""
        array = np.asarray(self.array)
        if np.issubdtype(array.dtype, np.unsignedinteger):
            # 8-bit WAV: unsigned, centred on 128, not on zero
            mid = (np.iinfo(array.dtype).max + 1) / 2
            array = (array.astype(np.float64) - mid) / mid
        elif np.issubdtype(array.dtype, np.integer):
            # PCM as the file or the microphone delivered it
            array = array / np.iinfo(array.dtype).max
        array = array.astype(np.float32, copy=False)
        if array.ndim == 2:
            # (samples, channels) as soundfile and Gradio lay it out,
            # (channels, samples) as torchaudio does: channels are the short
            # axis, there being far fewer of them than samples.
            if array.shape[0] > array.shape[1]:
                array = array.T
            if array.shape[0] == 1:
                array = array[0]  # one channel is mono
        if array.ndim not in (1, 2):
            raise ValueError(f"audio must be 1-D or 2-D, got shape {array.shape}")
        object.__setattr__(self, "array", np.ascontiguousarray(array))
        object.__setattr__(self, "rate", int(self.rate))
        if self.rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.rate}")

    @property
    def channels(self) -> int:
        """How many channels: 1 for a 1-D array."""
        return 1 if self.array.ndim == 1 else self.array.shape[0]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.array.shape[-1] / self.rate

    def mono(self, channel: int = 0) -> "Audio":
        """Return one channel as a 1-D :class:`Audio`; itself when already mono.

        The reference channel, not the mean: a microphone array's channels
        differ in delay, and their mean cancels what one channel keeps.

        Args:
            channel: Which channel, 0 by default.

        Examples:
            >>> Audio(np.zeros((2, 8000), dtype=np.float32), 8000).mono().array.shape
            (8000,)
        """
        if self.array.ndim == 1:
            return self
        return Audio(self.array[channel], self.rate)

    def multichannel(self) -> "Audio":
        """Return this audio as ``(channels, samples)``; mono becomes ``(1, samples)``.

        For a hook that wants one rank whatever came in.

        Examples:
            >>> Audio(np.zeros(8000, dtype=np.float32), 8000).multichannel().array.shape
            (1, 8000)
        """
        if self.array.ndim == 2:
            return self
        out = Audio.__new__(Audio)
        object.__setattr__(out, "array", self.array[None, :])
        object.__setattr__(out, "rate", self.rate)
        return out

    @classmethod
    def read(cls, path: str | Path, rate: int | None = None) -> "Audio":
        """Read an audio file, at its own rate or resampled to ``rate``.

        Args:
            path: Any file ``soundfile`` reads (WAV, FLAC, OGG, ...).
            rate: When given, the result is resampled to this rate.

        Returns:
            The file as float32, every channel kept.

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
            A new :class:`Audio` at ``rate``, every channel resampled, or
            this one when it already is.

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
            librosa.resample(self.array, orig_sr=self.rate, target_sr=rate, axis=-1),
            rate,
        )

    @classmethod
    def concat(cls, pieces: Sequence["Audio"]) -> "Audio":
        """Join consecutive pieces of one signal in time.

        Raises:
            ValueError: If the pieces differ in rate or in channel count.
        """
        rates = {p.rate for p in pieces}
        if len(rates) != 1:
            raise ValueError(f"cannot concatenate audio at rates {sorted(rates)}")
        channels = {p.channels for p in pieces}
        if len(channels) != 1:
            raise ValueError(
                f"cannot concatenate audio with {sorted(channels)} channels"
            )
        return cls(np.concatenate([p.array for p in pieces], axis=-1), rates.pop())

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
            The audio, resampled when its own rate differs, every channel
            kept.

        Raises:
            TypeError: If ``value`` is none of those, or carries no rate
                when none is fixed.

        Examples:
            >>> Audio.coerce("utt.wav", 16000).rate
            16000
            >>> Audio.coerce((44100, samples), 16000).rate    # from gr.Audio
            16000
            >>> Audio.coerce(torch.zeros(2, 16000), 16000).channels
            2
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


class AudioKind(Kind):
    """``audio``: an :class:`Audio` at the model's rate, with the field's channels.

    ``Field(..., channels=1)`` (the default) hands the hook the reference
    channel as a 1-D array; ``channels=None`` every channel as
    ``(channels, samples)``, mono included as one row; ``channels=N``
    exactly ``N`` channels, or a ``TypeError``. Pieces of a stream
    concatenate in time.
    """

    def check(self, value, field, model, *, output):
        """Coerce what a caller holds, then apply the field's channel count.

        Args:
            value: What the caller gave, or what the hook returned - a
                hook's bare array is taken to be at the model's rate.
            field: The declaration; its ``channels`` decides the shape.
            model: For its ``sample_rate``.
            output: Whether ``value`` is a hook's result.

        Returns:
            An :class:`Audio` at the model's rate: 1-D for one channel,
            ``(channels, samples)`` otherwise.

        Raises:
            TypeError: If ``value`` is not audio, carries no rate when the
                model fixes none, or has the wrong number of channels.

        Examples:
            >>> kind, stereo_in = AudioKind(), (16000, stereo)
            >>> mono_field = Field("speech", "audio")
            >>> kind.check(stereo_in, mono_field, model, output=False).array.ndim
            1
            >>> any_channels = Field("mix", "audio", channels=None)
            >>> kind.check(stereo_in, any_channels, model, output=False).channels
            2
            >>> two = Field("mix", "audio", channels=2)
            >>> kind.check((16000, mono), two, model, output=False)
            Traceback (most recent call last):
            TypeError: 'mix' given with 1 channel(s), needs 2
        """
        rate = model.sample_rate
        if output and isinstance(value, np.ndarray):
            if rate is None:
                raise TypeError(
                    f"{field.name!r} returned as a bare array by a model that "
                    "fixes no sample_rate; return an Audio with its rate"
                )
            audio = Audio(value, rate)
        else:
            audio = Audio.coerce(value, rate)
        want = getattr(field, "channels", 1)
        if want == 1:
            return audio.mono()
        if want is not None and audio.channels != want:
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} with {audio.channels} channel(s), needs {want}"
            )
        return audio.multichannel()

    def join(self, first, second):
        """Concatenate the samples in time."""
        return Audio.concat([first, second])

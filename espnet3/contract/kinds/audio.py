"""The ``audio`` kind: a waveform at a rate, mono or multichannel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

from espnet3.contract.kinds.base import Kind


@dataclass(frozen=True)
class Audio:
    """A float32 waveform and the rate it is sampled at.

    The one shape audio has once it is inside the API. Anything a caller is
    likely to hold - a path, the ``(rate, samples)`` pair ``gr.Audio``
    returns, a NumPy array, a torch tensor - becomes one of these through
    :meth:`coerce`, resampled to the rate the model wants, so a hook only
    ever sees ``audio.array`` at ``audio.rate``.

    Every channel is kept: ``array`` is ``(samples,)`` for mono and
    ``(channels, samples)`` otherwise. ``Field.channels`` can constrain the
    count; explicit normalizers select or average channels.

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
        """Count the channels: 1 for a 1-D array.

        Returns:
            The number of channels.

        Examples:
            >>> Audio(np.zeros((2, 8000), dtype=np.float32), 8000).channels
            2
        """
        return 1 if self.array.ndim == 1 else self.array.shape[0]

    @property
    def duration(self) -> float:
        """Measure the duration in seconds.

        Returns:
            The number of samples divided by the sample rate.

        Examples:
            >>> Audio(np.zeros(8000, dtype=np.float32), 8000).duration
            1.0
        """
        return self.array.shape[-1] / self.rate

    def mono(self, channel: int = 0) -> "Audio":
        """Return one channel as a 1-D :class:`Audio`; itself when already mono.

        The reference channel, not the mean: a microphone array's channels
        differ in delay, and their mean cancels what one channel keeps.

        Args:
            channel: Which channel, 0 by default.

        Returns:
            The selected channel as a 1-D ``Audio``, or ``self`` when mono.

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

        Returns:
            The audio as ``(channels, samples)``; ``self`` when already 2-D.

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

        Args:
            pieces: The consecutive pieces, all at one rate with one channel count.

        Returns:
            One ``Audio`` holding the pieces back to back.

        Raises:
            ValueError: If the pieces differ in rate or in channel count.

        Examples:
            >>> piece = Audio(np.zeros(8000, dtype=np.float32), 8000)
            >>> Audio.concat([piece, piece]).duration
            2.0
        """
        rates = {p.rate for p in pieces}
        if len(rates) != 1:
            raise ValueError(f"cannot concatenate audio at rates {sorted(rates)}")
        channels = {p.channels for p in pieces}
        if len(channels) != 1:
            raise ValueError(
                f"cannot concatenate audio with {sorted(channels)} channels"
            )
        # one channel may be laid out as (samples,) or as (1, samples) - a
        # mono piece becomes the former on construction, a field asking for
        # every channel hands out the latter - so the pieces are brought to
        # one layout before they are joined
        arrays = [p.multichannel().array for p in pieces]
        return cls(np.concatenate(arrays, axis=-1), rates.pop())

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

    Every channel and the sample rate are preserved by default. ``channels=N``
    validates the channel count. Use ``first_channel`` or ``mean_channels``
    normalizers to explicitly convert multichannel audio to mono.
    """

    normalizers = {
        "first_channel": lambda audio: audio.mono(),
        "mean_channels": lambda audio: Audio(
            audio.array.mean(axis=0) if audio.array.ndim == 2 else audio.array,
            audio.rate,
        ),
    }

    option_names = Kind.option_names | {"sample_rate", "rate_policy"}

    def validate_options(self, options) -> None:
        """Validate the target sample rate and mismatch policy.

        Args:
            options: The field's options: ``sample_rate``, ``rate_policy`` and
                ``normalize``.

        Returns:
            None.

        Raises:
            ValueError: If ``sample_rate`` is not a positive integer or
                ``rate_policy`` is not ``resample`` or ``reject``.

        Examples:
            >>> AudioKind().validate_options({"sample_rate": 16000})
            >>> AudioKind().validate_options({"rate_policy": "keep"})
            Traceback (most recent call last):
            ValueError: rate_policy must be 'resample' or 'reject'
        """
        super().validate_options(options)
        rate = options.get("sample_rate")
        if rate is not None and (
            isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")
        if options.get("rate_policy", "resample") not in ("resample", "reject"):
            raise ValueError("rate_policy must be 'resample' or 'reject'")

    def check_configured(self, value, field, model, options, *, output):
        """Apply the field's rate policy before the standard audio check.

        Args:
            value: Audio in any form :meth:`Audio.coerce` accepts.
            field: The declaration the value is for.
            model: The instance; its ``sample_rate`` is the default rate.
            options: The field's options: ``sample_rate`` and ``rate_policy``.
            output: Whether the value is a hook's result.

        Returns:
            An ``Audio`` at the target rate.

        Raises:
            ValueError: If the policy is ``reject`` and the rates differ.

        Examples:
            >>> from espnet3.contract import Field
            >>> value = (16000, np.zeros(16000, dtype=np.float32))
            >>> audio = AudioKind().check_configured(
            ...     value, Field("speech", "audio"), None,
            ...     {"sample_rate": 16000}, output=False,
            ... )
            >>> audio.rate
            16000
        """
        rate = options.get("sample_rate")
        if rate is None:
            rate = getattr(model, "sample_rate", None)
        if options.get("rate_policy", "resample") == "reject":
            source_rate = None
            if isinstance(value, Audio):
                source_rate = value.rate
            elif isinstance(value, (str, Path)):
                value = Audio.read(value)
                source_rate = value.rate
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                if isinstance(value[0], (int, np.integer)):
                    source_rate = value[0]
            if source_rate is not None and rate is not None and source_rate != rate:
                raise ValueError(
                    f"{field.name!r} has sample rate {source_rate}; expected {rate}"
                )
        return self.check(
            value, field, SimpleNamespace(sample_rate=rate), output=output
        )

    def to_demo_output(self, value):
        """Return Audio as (rate, samples), with channels last for Gradio.

        Args:
            value: A checked output; anything but ``Audio`` is returned as is.

        Returns:
            A ``(rate, samples)`` pair for an ``Audio``, else ``value``.

        Examples:
            >>> audio = Audio(np.zeros((2, 8000), dtype=np.float32), 8000)
            >>> rate, samples = AudioKind().to_demo_output(audio)
            >>> rate, samples.shape
            (8000, (8000, 2))
        """
        if not isinstance(value, Audio):
            return value
        array = value.array
        return value.rate, array.T if array.ndim == 2 else array

    def check(self, value, field, model, *, output):
        """Coerce what a caller holds, then validate the field's channel count.

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
            2
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
        want = field.channels
        if want is not None and audio.channels != want:
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} with {audio.channels} channel(s), needs {want}"
            )
        return audio

    def join(self, first, second):
        """Concatenate two pieces of audio in time.

        Args:
            first: The audio gathered so far.
            second: The audio that follows it.

        Returns:
            The two as one ``Audio``.

        Examples:
            >>> piece = Audio(np.zeros(8000, dtype=np.float32), 8000)
            >>> AudioKind().join(piece, piece).duration
            2.0
        """
        return Audio.concat([first, second])

    def accepts(self, value, field):
        """Accept anything :meth:`check` would, plus a bare rate-less array.

        :meth:`check` always needs ``model.sample_rate``, which a model-free
        caller cannot supply, so the default ``accepts`` (which calls
        ``check`` with ``model=None``) would reject every value, including
        an ``Audio``, a path or a ``(rate, samples)`` pair that already
        carries its own rate. This instead accepts any form
        :meth:`Audio.coerce` accepts when given a fixed rate, without
        requiring one: an :class:`Audio`, a path, a ``(rate, samples)``
        pair, or a bare array/tensor.

        Args:
            value: The value to test.
            field: The declaration the value is for.

        Returns:
            ``True`` when ``value`` could be audio, ``False`` otherwise.

        Examples:
            >>> from espnet3.contract import Field
            >>> field = Field("speech", "audio")
            >>> AudioKind().accepts(np.zeros(16000, dtype=np.float32), field)
            True
            >>> AudioKind().accepts((16000, np.zeros(16000, dtype=np.float32)), field)
            True
            >>> AudioKind().accepts(42, field)
            False
        """
        if isinstance(value, Audio):
            return True
        if isinstance(value, (str, Path)):
            return True
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and isinstance(value[0], (int, np.integer))
        ):
            return True
        if hasattr(value, "detach"):  # a torch tensor, without importing torch
            return True
        return isinstance(value, np.ndarray)

"""The inference contract: what a trained ESPnet3 model promises a caller.

Every system under ``espnet3/systems/<name>/`` ships an ``inference.py``
whose ``Inference`` class subclasses :class:`InferenceAPI`. The class
declares the fields it takes and returns and implements one hook -
:meth:`InferenceAPI.run_stream` if the model works online,
:meth:`InferenceAPI.run` if it needs its whole input - and the base class
does everything a caller should not have to think about: reading a file,
accepting what Gradio hands over, resampling to the model's rate, checking
the declared fields, deriving the other hook. So the command line, the MCP
server, a Space, a notebook and the ``infer`` stage can all drive any
system the same way.

Three ways to call a loaded model::

    >>> from espnet3.api.inference import load
    >>> model = load("espnet/some_pack")           # meta.yaml names the system
    >>> model("utt.wav")["text"]                   # one shot, from a file
    >>> model(speech=(16000, samples))["text"]     # one shot, from gr.Audio
    >>> for piece in model.stream(microphone_chunks()):
    ...     print(piece.get("text", ""), end="")   # online
    >>> model.batch([{"speech": a}, {"speech": b}])  # several at once

Inference is a stream of chunks in and chunks out; the one-shot call is
the stream of one chunk, and a batch is several one-shot calls that a
model may choose to run together.

What a field can hold is a :class:`Kind` registered in :data:`KINDS`;
``audio``, ``text`` and ``segments`` are built in, and a new modality is
one registered subclass.

A system says nothing about *what task* it performs; it says what goes in
and what comes out. A front end that offers ``transcribe`` looks for a
model whose first input is audio and whose outputs hold ``text``, and a
model that answers a conversation declares one ``messages`` field and no
list of the tasks a prompt might ask of it.

Relation to the provider and runner
-----------------------------------

ESPnet3 runs its stages through an ``EnvironmentProvider`` and a
``BaseRunner`` (``espnet3/parallel``): the provider builds the objects a
stage needs, the runner processes dataset shards, in parallel, with
resume and writers. The contract and that pair divide the work like this:

1. ``Inference`` is the only thing a system must provide for inference.
   It knows nothing of datasets, shards or runners, so the front ends -
   which have none - can use it, and so can a test.
2. The ``infer`` stage runs an ``Inference`` through the pair, and needs
   nothing added to it: ``inference.yaml``'s ``model._target_`` names the
   class (its constructor takes the model's own arguments, plus
   ``device``), ``InferenceProvider.build_model`` instantiates it, and
   ``InferenceRunner`` calls ``model(**fields)`` for one item or with a
   list per field for a batch, and writes the mapping it returns. The
   recipe's ``output_fn`` stays optional, for columns the contract does
   not produce, such as ``ref`` for scoring.
3. Parallelism - shards, workers, resume, writers - belongs to the
   runner. Decoding several items together belongs to :meth:`run_batch`.
   Streaming belongs to :meth:`run_stream`; the runner never streams.
4. System authors do not subclass the provider or runner to implement
   inference. They subclass them only for what the pair is for: how a
   dataset is built, or how outputs are written.
5. A system may not use the pair at all - a SpeechLM served by vLLM, a
   model behind an endpoint - and still provides ``Inference``, whose
   ``from_pretrained`` takes whatever handle it needs (a URL, a name),
   given to :func:`load` with ``system=`` since there is no bundle. It
   need not have an ``infer`` stage config; if it has one, it may run it
   its own way. The front ends only ever see ``Inference``.
6. Nothing assumes a task. No layer - the contract, a kind, the
   provider/runner, a front end - may ask a system what task it performs
   or dispatch on a task name; what a model does is read from its fields.
   A verb such as ``transcribe`` is a front end's word for a shape of
   fields, and a multi-task model declares its fields once.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class Field:
    """One input or output of a system.

    A system lists these in its ``inputs`` and ``outputs``; the base class
    binds arguments to them, converts and checks values by ``kind``, and a
    front end builds its widgets or command-line arguments from them.

    Args:
        name: The keyword the value is passed or returned as, such as
            ``speech`` or ``text``. Must be a Python identifier.
        kind: A name in :data:`KINDS`. Decides how a value is converted
            and checked (``audio`` becomes an :class:`Audio` at the model's
            rate, ``text`` must be a ``str``) and, for a front end, which
            widget or argument type shows it.
        label: What a page calls the field. Defaults to the name with
            underscores spaced and the first letter capitalised.
        optional: An input the caller may leave out; the hook then does
            not receive it. Outputs are never optional.

    Raises:
        ValueError: If ``kind`` is not registered in :data:`KINDS` or
            ``name`` is not an identifier.

    Examples:
        >>> Field("speech", "audio")
        Field(name='speech', kind='audio', label='Speech', optional=False)
        >>> Field("reference_speech", "audio", optional=True).label
        'Reference speech'
        >>> Field("text", "text", "Transcription")
        Field(name='text', kind='text', label='Transcription', optional=False)
    """

    name: str
    kind: str
    label: str = ""
    optional: bool = False

    def __post_init__(self) -> None:
        """Check the kind and the name, and fill in the label."""
        if self.kind not in KINDS:
            raise ValueError(
                f"Field {self.name!r} has kind {self.kind!r}; "
                f"known kinds are {sorted(KINDS)}"
            )
        if not self.name.isidentifier():
            raise ValueError(f"Field name {self.name!r} must be a Python identifier")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").capitalize())


@dataclass(frozen=True)
class Audio:
    """A mono float32 waveform and the rate it is sampled at.

    The one shape audio has once it is inside the API. Anything a caller is
    likely to hold - a path, the ``(rate, samples)`` pair ``gr.Audio``
    returns, a NumPy array, a torch tensor - becomes one of these through
    :meth:`coerce`, resampled to the rate the model wants, so a hook only
    ever sees ``audio.array`` at ``audio.rate``.

    Args:
        array: The samples. Integer PCM is scaled to ``[-1, 1]``; a 2-D
            array is averaged to mono, taking the shorter axis as channels
            (``(samples, channels)`` from soundfile and Gradio,
            ``(channels, samples)`` from torchaudio).
        rate: Samples per second.

    Raises:
        ValueError: If the array is not 1-D or 2-D, or the rate is not
            positive.

    Examples:
        >>> Audio(np.zeros(16000, dtype=np.float32), 16000).seconds
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
            # axis, there being far fewer of them than samples.
            array = array.mean(axis=0 if array.shape[0] < array.shape[1] else 1)
        if array.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {array.shape}")
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "rate", int(self.rate))
        if self.rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.rate}")

    @property
    def seconds(self) -> float:
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
        return audio if rate is None else audio.to(rate)

    def to(self, rate: int) -> "Audio":
        """Return this audio at ``rate``; itself when already there."""
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
    def coerce(cls, value: Any, rate: int) -> "Audio":
        """Turn whatever a caller holds into an :class:`Audio` at ``rate``.

        Args:
            value: One of: an :class:`Audio`; a path, which is read; a
                ``(rate, samples)`` pair, which is what ``gr.Audio`` returns;
                a NumPy array or a torch tensor, taken to be at ``rate``
                already, because nothing says otherwise.
            rate: The rate the result is at.

        Returns:
            The audio, resampled when its own rate differs.

        Raises:
            TypeError: If ``value`` is none of those.

        Examples:
            >>> Audio.coerce("utt.wav", 16000).rate
            16000
            >>> Audio.coerce((44100, samples), 16000).rate    # from gr.Audio
            16000
            >>> Audio.coerce(torch.zeros(1, 16000), 16000).seconds
            1.0
        """
        if isinstance(value, Audio):
            return value.to(rate)
        if isinstance(value, (str, Path)):
            return cls.read(value, rate)
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and isinstance(value[0], (int, np.integer))
        ):
            return cls(np.asarray(value[1]), int(value[0])).to(rate)
        if hasattr(value, "detach"):  # a torch tensor, without importing torch
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return cls(value, rate)
        raise TypeError(
            "audio must be a path, an Audio, a (rate, samples) pair, "
            f"an array or a tensor, not {type(value).__name__}"
        )


class Kind(ABC):
    """What a field's ``kind`` means: how a value is converted, checked and joined.

    A kind turns what a caller gives into what a hook receives, checks
    what a hook returns, and says how pieces of a stream join. The
    built-in kinds are ``audio``, ``text`` and ``segments``; a new
    modality - a conversation, a multichannel signal, video, a JSON
    document - is a subclass registered in :data:`KINDS`, and needs no
    change to :class:`InferenceAPI`.

    Examples:
        >>> class Messages(Kind):
        ...     def check(self, value, field, model, *, output):
        ...         if not isinstance(value, list):
        ...             raise TypeError(f"{field.name} must be a list of turns")
        ...         return value
        >>> KINDS["messages"] = Messages()
        >>> Field("messages", "messages")
        Field(name='messages', kind='messages', label='Messages', optional=False)
    """

    @abstractmethod
    def check(
        self, value: Any, field: Field, model: "InferenceAPI", *, output: bool
    ) -> Any:
        """Return ``value`` converted for ``field``, or raise ``TypeError``.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration the value is for.
            model: The instance, for what it knows (its ``sample_rate``).
            output: Whether the value is a hook's result rather than a
                caller's argument; a kind may accept different forms for
                each.
        """

    def join(self, first: Any, second: Any) -> Any:
        """Join two consecutive pieces of one field; the default is ``+``."""
        return first + second

    def is_batch(self, value: Any, field: Field, model: "InferenceAPI") -> bool:
        """Tell one entry per sample from one value of this kind.

        A list that is not itself a valid value of the kind is a batch:
        a list of arrays is several utterances, a ``[rate, samples]`` pair
        is one; a list of turns is one conversation, a list of those is
        several. Override when a kind can say it more cheaply.
        """
        if not isinstance(value, list):
            return False
        try:
            self.check(value, field, model, output=False)
        except TypeError:
            return True
        return False


class AudioKind(Kind):
    """``audio``: an :class:`Audio` at the model's rate; pieces concatenate."""

    def check(self, value, field, model, *, output):
        """Coerce what a caller holds; a hook's bare array is at the model's rate."""
        if output and isinstance(value, np.ndarray):
            return Audio(value, model.sample_rate)
        return Audio.coerce(value, model.sample_rate)

    def join(self, first, second):
        """Concatenate the samples."""
        return Audio.concat([first, second])


class TextKind(Kind):
    """``text``: a ``str``; pieces append."""

    def check(self, value, field, model, *, output):
        """Require a str."""
        if not isinstance(value, str):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, must be str"
            )
        return value


class SegmentsKind(Kind):
    """``segments``: dicts with ``text``, ``start`` and ``end`` in seconds.

    The shape ``espnet align`` prints; ``score`` is optional. Pieces append.
    """

    def check(self, value, field, model, *, output):
        """Require a list of dicts with the three keys."""
        if not isinstance(value, list) or not all(
            isinstance(s, Mapping) and {"text", "start", "end"} <= set(s) for s in value
        ):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} must be a list of dicts "
                "with text, start and end"
            )
        return value


# What a field can hold, by the name a Field's ``kind`` gives. Register a
# subclass of Kind here to add a modality.
KINDS: dict[str, Kind] = {
    "audio": AudioKind(),
    "text": TextKind(),
    "segments": SegmentsKind(),
}


def check_contract(cls: type) -> None:
    """Raise ``TypeError`` unless ``cls`` is a usable system declaration.

    Run on every concrete subclass of :class:`InferenceAPI` as it is
    defined, so a system that gets the declaration wrong fails at import,
    not in a Space at runtime. The rules:

    - ``inputs`` and ``outputs`` are non-empty tuples of :class:`Field`
      with distinct names;
    - required inputs come before optional ones, so a call by position
      means one thing;
    - no output is optional;
    - :meth:`InferenceAPI.run_stream` or :meth:`InferenceAPI.run` is
      implemented, since each is the other's default.

    Args:
        cls: The class to check.

    Raises:
        TypeError: With the rule that was broken.

    Examples:
        >>> class Bad(InferenceAPI):
        ...     inputs = (Field("prompt", "text", optional=True),
        ...               Field("speech", "audio"))
        ...     outputs = (Field("text", "text"),)
        Traceback (most recent call last):
        TypeError: Bad.inputs must list required fields before optional ones, ...
    """
    for attr in ("inputs", "outputs"):
        fields = getattr(cls, attr, None)
        if not isinstance(fields, tuple) or not all(
            isinstance(f, Field) for f in fields
        ):
            raise TypeError(f"{cls.__qualname__}.{attr} must be a tuple of Field")
        if not fields:
            raise TypeError(f"{cls.__qualname__}.{attr} must name at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise TypeError(f"{cls.__qualname__}.{attr} repeats a name: {names}")
    optional = [f.optional for f in cls.inputs]
    if optional != sorted(optional):
        raise TypeError(
            f"{cls.__qualname__}.inputs must list required fields before "
            "optional ones, so a call by position means one thing"
        )
    for f in cls.outputs:
        if f.optional:
            raise TypeError(f"{cls.__qualname__}: outputs cannot be optional")
    if cls.run is InferenceAPI.run and cls.run_stream is InferenceAPI.run_stream:
        raise TypeError(
            f"{cls.__qualname__} must implement run_stream (online) or run "
            "(the whole input at once); each is the other's default"
        )


class InferenceAPI(ABC):
    """What a system implements to be callable from every front end.

    A subclass declares two class attributes and implements three members:

    - ``inputs`` / ``outputs``: tuples of :class:`Field`; required inputs
      first, so a call by position means one thing.
    - :meth:`from_pretrained`: build one from a packed model directory or a
      Hub tag.
    - :attr:`sample_rate`: the rate the model takes audio at.
    - one of :meth:`run_stream` (online) and :meth:`run` (the whole input
      at once); the base class derives the other. :meth:`run_batch` may be
      overridden when the model decodes several inputs faster together.

    Callers use the entry points, never the hooks: the one-shot call
    ``model(...)``, :meth:`stream` and :meth:`batch`. They bind arguments
    to the declared fields, convert and check every value, and check what
    comes back, so a hook sees :class:`Audio` at :attr:`sample_rate` for
    every audio field, ``str`` for every text field, and never a required
    field missing.

    Notes:
        ``model(**fields)`` returning a mapping of the declared outputs is
        exactly what the ``infer`` stage's ``InferenceRunner`` expects of a
        model, and a list per field is how that runner passes a batch; so
        an instance serves the recipe's ``infer`` stage as it is, with
        ``input_key`` naming the dataset fields and no ``output_fn``.

    Examples:
        A system whose model needs the whole input::

            class Inference(InferenceAPI):
                inputs = (Field("speech", "audio"),)
                outputs = (Field("text", "text"),)

                @classmethod
                def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
                    return cls(load_backend(locate_pack(tag_or_dir), device=device))

                @property
                def sample_rate(self):
                    return 16000

                def run(self, speech):
                    return {"text": self.backend(speech.array)[0][0]}

        A system whose model works online yields as it goes; the one-shot
        call then gathers what it yields::

            class Inference(InferenceAPI):
                inputs = (Field("speech", "audio"),)
                outputs = (Field("text", "text"),)
                ...
                def run_stream(self, chunks):
                    for chunk in chunks:
                        if "speech" in chunk:
                            yield {"text": self.decoder.feed(chunk["speech"].array)}
                    yield {"text": self.decoder.finish()}

        Using either::

            >>> model = Inference.from_pretrained("exp/model_pack")
            >>> model("utt.wav")
            {'text': 'hello world'}
            >>> list(model.stream([{"speech": first_half}, {"speech": second_half}]))
            [{'text': 'hello '}, {'text': 'world'}]
            >>> model.batch([{"speech": "a.wav"}, {"speech": "b.wav"}])
            [{'text': '...'}, {'text': '...'}]
    """

    inputs: ClassVar[tuple[Field, ...]]
    outputs: ClassVar[tuple[Field, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check a system's declaration the moment its class is defined."""
        super().__init_subclass__(**kwargs)
        # An intermediate base that declares nothing is allowed; a system
        # declares both.
        if hasattr(cls, "inputs") or hasattr(cls, "outputs"):
            check_contract(cls)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, tag_or_dir: str | Path, *, device: str = "cpu", **kwargs: Any
    ) -> "InferenceAPI":
        """Load a published model.

        Args:
            tag_or_dir: A ``pack_model`` output directory, or a Hub tag
                ``espnet_model_zoo`` resolves. :func:`locate_pack` turns
                either into the directory.
            device: Where to build the model, ``"cpu"`` or ``"cuda:0"``.
            **kwargs: Whatever the system needs beyond that; a system that
                needs nothing rejects any.

        Returns:
            A ready instance.

        Examples:
            >>> Inference.from_pretrained("exp/model_pack")
            >>> Inference.from_pretrained("espnet/some_pack", device="cuda:0")
        """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The rate every audio input is resampled to before a hook sees it."""

    # -- the hooks a system implements; each has the other as its default --

    def run_stream(self, chunks: Iterable[Mapping[str, Any]]) -> Iterator[Mapping]:
        """Consume input chunks as they arrive; yield output chunks when ready.

        The hook for a model that works online. A chunk is a mapping of
        field name to a piece of that field: a slice of audio, more text,
        some segments. Not every chunk carries every field; a hook may yield
        nothing for one chunk and several outputs for another, and may
        yield after the input ends. The input ends when the iterable does.

        Args:
            chunks: Input chunks, already converted and checked: audio
                pieces are :class:`Audio` at :attr:`sample_rate`.

        Yields:
            Output chunks, mappings of output field name to a piece of it.
            **A piece is what is new since the last chunk, never the state
            so far**: pieces of one field are joined by :func:`gather` -
            audio concatenated, text and segments appended - so a
            cumulative transcript would come out doubled. A streaming ASR
            model whose decoder holds ``"hello"`` and then ``"hello world"``
            yields ``{"text": "hello"}`` and then ``{"text": " world"}``.

        Notes:
            The default gathers every chunk and calls :meth:`run` once,
            which is what a model that needs its whole input does.

        Examples:
            A blockwise decoder, one piece per chunk and a flush at the end::

                def run_stream(self, chunks):
                    decoder = self.new_decoder()
                    committed = ""
                    for chunk in chunks:
                        text = decoder.feed(chunk["speech"].array)  # cumulative
                        yield {"text": text[len(committed):]}
                        committed = text
                    yield {"text": decoder.finish()[len(committed):]}
        """
        yield self.run(**gather(self.inputs, chunks))

    def run(self, **inputs: Any) -> Mapping[str, Any]:
        """Infer one complete sample.

        The hook for a model that needs its whole input.

        Args:
            **inputs: The declared input fields, converted and checked;
                optional ones the caller left out are absent.

        Returns:
            A mapping holding every declared output; extra keys pass
            through to the caller.

        Notes:
            The default is the stream of one chunk, gathered: what a
            streaming model does when handed everything at once.
        """
        return gather(self.outputs, self.run_stream(iter([inputs])))

    def run_batch(self, items: Sequence[Mapping[str, Any]]) -> Sequence[Mapping]:
        """Infer several complete samples.

        Override when the model decodes a batch faster than one at a time;
        the default runs :meth:`run` on each item.

        Args:
            items: One mapping of converted inputs per sample.

        Returns:
            One output mapping per item, in order.
        """
        return [self.run(**item) for item in items]

    # -- the entry points a caller uses --

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Infer one complete sample, or a batch given as lists.

        Positional values fill ``inputs`` in order, so ``model(audio)`` is
        ``model(speech=audio)`` for any model whose first input is audio.
        When every given value is a list that is not itself a value of its
        kind (see :meth:`Kind.is_batch`), they are a batch - one entry per
        sample, as ``InferenceRunner`` passes one - and the result is a
        list of outputs, as from :meth:`batch`.

        Returns:
            The declared outputs, converted and checked, plus whatever else
            the hook returned; a list of those for a batch.

        Raises:
            TypeError: For a missing required input, an unknown one, one
                given twice, or a value of the wrong kind.
            RuntimeError: If the hook left out a declared output.

        Examples:
            >>> model("utt.wav")
            {'text': 'hello world'}
            >>> model(speech=(16000, samples))
            {'text': 'hello world'}
            >>> model(speech=[first, second])
            [{'text': '...'}, {'text': '...'}]
        """
        values = self._collect(args, kwargs)
        fields = {f.name: f for f in self.inputs}
        if values and all(
            KINDS[fields[name].kind].is_batch(v, fields[name], self)
            for name, v in values.items()
        ):
            lengths = {len(v) for v in values.values()}
            if len(lengths) != 1:
                raise TypeError(f"batch inputs differ in length: {sorted(lengths)}")
            return self.batch([dict(zip(values, row)) for row in zip(*values.values())])
        bound = self._bind(values)
        return self._check_output(self.run(**bound), complete=True)

    def stream(self, chunks: Iterable[Mapping[str, Any]]) -> Iterator[dict]:
        """Infer from a stream of input chunks, yielding output chunks.

        Each input chunk is checked and converted as a one-shot call's
        arguments are, field by field; each output chunk likewise, though
        an output chunk need not hold every field.

        Args:
            chunks: Mappings of input field name to a piece of it. A lazy
                iterable - a microphone, a socket - works: chunks are pulled
                as the hook asks for them.

        Yields:
            Output chunks, converted and checked.

        Raises:
            TypeError: For an unknown field or a value of the wrong kind,
                and, once the input ends, for a required input that never
                arrived.

        Examples:
            >>> for out in model.stream({"speech": a} for a in microphone()):
            ...     print(out.get("text", ""), end="", flush=True)
        """
        seen: set[str] = set()

        def checked() -> Iterator[dict[str, Any]]:
            for chunk in chunks:
                piece = self._bind(self._collect((), dict(chunk)), partial=True)
                seen.update(piece)
                yield piece
            missing = [
                f.name for f in self.inputs if not f.optional and f.name not in seen
            ]
            if missing:
                raise TypeError(f"{type(self).__qualname__} never got {missing}")

        for out in self.run_stream(checked()):
            yield self._check_output(out, complete=False)

    def batch(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Infer several complete samples, converting and checking each.

        Args:
            items: One mapping of inputs per sample, each as a one-shot
                call would take them.

        Returns:
            One output mapping per item, in order.

        Raises:
            RuntimeError: If :meth:`run_batch` returned a different number
                of outputs than items.

        Examples:
            >>> model.batch([{"speech": "a.wav"}, {"speech": "b.wav"}])
            [{'text': '...'}, {'text': '...'}]
        """
        bound = [self._bind(self._collect((), dict(item))) for item in items]
        outputs = list(self.run_batch(bound))
        if len(outputs) != len(items):
            raise RuntimeError(
                f"{type(self).__qualname__}.run_batch returned {len(outputs)} "
                f"outputs for {len(items)} items"
            )
        return [self._check_output(out, complete=True) for out in outputs]

    # -- checking --

    def _collect(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge positional and keyword arguments into one name-keyed dict."""
        names = [f.name for f in self.inputs]
        if len(args) > len(names):
            raise TypeError(
                f"{type(self).__qualname__} takes at most {len(names)} "
                f"positional inputs {names}, got {len(args)}"
            )
        values = dict(kwargs)
        for name, value in zip(names, args):
            if name in values:
                raise TypeError(f"{name!r} given both by position and by name")
            values[name] = value
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise TypeError(
                f"{type(self).__qualname__} has no input {unknown}; inputs are {names}"
            )
        return values

    def _bind(self, values: dict[str, Any], *, partial: bool = False) -> dict:
        """Convert and check the given values; ``partial`` allows any to be absent."""
        bound: dict[str, Any] = {}
        for f in self.inputs:
            value = values.get(f.name)
            if value is None:
                if f.optional or partial:
                    continue
                raise TypeError(f"{type(self).__qualname__} needs {f.name!r}")
            bound[f.name] = self._check(f, value, output=False)
        return bound

    def _check_output(self, result: Any, *, complete: bool) -> dict[str, Any]:
        """Check a hook's result; ``complete`` demands every declared output."""
        if not isinstance(result, Mapping):
            raise TypeError(
                f"{type(self).__qualname__} produced "
                f"{type(result).__name__}, not a mapping"
            )
        out: dict[str, Any] = {}
        for f in self.outputs:
            if f.name not in result:
                if complete:
                    raise RuntimeError(
                        f"{type(self).__qualname__} did not return {f.name!r}"
                    )
                continue
            out[f.name] = self._check(f, result[f.name], output=True)
        for name, value in result.items():
            out.setdefault(name, value)
        return out

    def _check(self, f: Field, value: Any, *, output: bool) -> Any:
        """Convert and check one value by its field's kind."""
        return KINDS[f.kind].check(value, f, self, output=output)


def gather(fields: tuple[Field, ...], chunks: Iterable[Mapping[str, Any]]) -> dict:
    """Join a stream of chunks into the one value each field would have had.

    This is what makes a one-shot call the special case of a stream, in
    both directions: the default :meth:`InferenceAPI.run_stream` gathers
    the input for :meth:`InferenceAPI.run`, and the default ``run`` gathers
    what ``run_stream`` yields.

    Args:
        fields: The declarations that say how each name joins, through
            its kind's :meth:`Kind.join`: audio concatenated, text and
            segments appended.
        chunks: The pieces, in order. A name outside ``fields`` keeps its
            last value.

    Returns:
        One value per name seen.

    Examples:
        >>> gather((Field("text", "text"),), [{"text": "hel"}, {"text": "lo"}])
        {'text': 'hello'}
    """
    kinds = {f.name: KINDS[f.kind] for f in fields}
    acc: dict[str, Any] = {}
    for chunk in chunks:
        for name, piece in chunk.items():
            if name not in acc or name not in kinds:
                acc[name] = piece
            else:
                acc[name] = kinds[name].join(acc[name], piece)
    return acc


# A system renamed after bundles were published under its old name: old name
# to current directory. ``load`` looks the name in ``meta.yaml`` up here, so a
# rename is one row and every bundle already on the Hub keeps loading.
SYSTEM_ALIASES: dict[str, str] = {}


def locate_pack(tag_or_dir: str | Path) -> Path:
    """Return the directory of a ``pack_model`` bundle, downloading a Hub tag.

    Args:
        tag_or_dir: An existing directory, returned resolved; or a tag
            ``espnet_model_zoo`` downloads and unpacks.

    Returns:
        The directory holding ``meta.yaml``.

    Raises:
        RuntimeError: If the download holds no ``inference_config``, so is
            not a ``pack_model`` bundle.

    Examples:
        >>> locate_pack("exp/train/model_pack")
        PosixPath('/.../exp/train/model_pack')
        >>> locate_pack("espnet/some_pack")   # downloaded into the cache
        PosixPath('/.../.cache/espnet/.../model_pack')
    """
    path = Path(tag_or_dir)
    if path.is_dir():
        return path.resolve()
    from espnet_model_zoo.downloader import ModelDownloader

    artifacts = ModelDownloader().download_and_unpack(str(tag_or_dir))
    if "inference_config" not in artifacts:
        raise RuntimeError(
            f"{tag_or_dir} is not a pack_model bundle: it has no inference_config"
        )
    return Path(artifacts["inference_config"]).resolve().parent.parent


def load(
    tag_or_dir: str | Path,
    *,
    device: str = "cpu",
    system: str | None = None,
    **kwargs: Any,
) -> InferenceAPI:
    """Load a published model behind its system's :class:`InferenceAPI`.

    The bundle's ``meta.yaml`` names the system that trained it
    (``system: esp2_asr``, written by ``pack_model``); this imports
    ``espnet3.systems.<system>.inference`` and calls its
    ``Inference.from_pretrained``. The caller needs to know nothing about
    the system.

    Args:
        tag_or_dir: A ``pack_model`` directory or a Hub tag.
        device: Where to build the model.
        system: Overrides the name in ``meta.yaml`` - for a bundle packed
            before the name was recorded, or a model that is not an ESPnet3
            bundle at all, in which case ``tag_or_dir`` is passed to the
            system as it is. A name a system has since given up is followed
            through :data:`SYSTEM_ALIASES`.
        **kwargs: Forwarded to ``from_pretrained``.

    Returns:
        The system's ``Inference`` instance.

    Raises:
        ValueError: If ``meta.yaml`` names no system and none is given.
        ImportError: If the system has no ``inference.Inference``.

    Examples:
        >>> model = load("espnet/some_pack")
        >>> model = load("exp/train/model_pack", device="cuda:0")
        >>> model = load("exp/old_pack", system="esp2_asr")   # older meta.yaml
    """
    if system is None:
        tag_or_dir = locate_pack(tag_or_dir)
        meta = yaml.safe_load((tag_or_dir / "meta.yaml").read_text("utf-8")) or {}
        system = meta.get("system")
        if not system:
            raise ValueError(
                f"{tag_or_dir}/meta.yaml does not name its system; it was packed "
                "before pack_model recorded one. Pass system=<name>."
            )
    system = SYSTEM_ALIASES.get(system, system)
    name = f"espnet3.systems.{system}.inference"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        # the system's own module missing is one thing; a dependency it
        # imports missing is another, and stays the error it was
        if e.name is None or not (e.name == name or name.startswith(e.name + ".")):
            raise
        raise ImportError(
            f"no {name}: system {system!r} has no Inference yet, or meta.yaml "
            "names the wrong system. Pass system=<name>."
        ) from e
    cls = getattr(module, "Inference", None)
    if not isinstance(cls, type) or not issubclass(cls, InferenceAPI):
        raise ImportError(
            f"espnet3.systems.{system}.inference defines no Inference(InferenceAPI)"
        )
    return cls.from_pretrained(tag_or_dir, device=device, **kwargs)

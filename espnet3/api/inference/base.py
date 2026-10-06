"""The contract itself: :class:`InferenceAPI`, its check, and :func:`gather`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Optional, Sequence

from espnet3.api.inference.field import Field
from espnet3.api.inference.kinds import KINDS
from espnet3.components.contract.check import check_declaration


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
    check_declaration(cls)
    if cls.run is InferenceAPI.run and cls.run_stream is InferenceAPI.run_stream:
        raise TypeError(
            f"{cls.__qualname__} must implement run_stream (online) or run "
            "(the whole input at once); each is the other's default"
        )


class InferenceAPI(ABC):
    """What a system implements to be callable from every front end.

    A subclass declares two class attributes and implements two members:

    - ``inputs`` / ``outputs``: tuples of :class:`Field`; required inputs
      first, so a call by position means one thing.
    - :meth:`from_pretrained`: build one from a packed model directory or a
      Hub tag.
    - one of :meth:`run_stream` (online) and :meth:`run` (the whole input
      at once); the base class derives the other. :meth:`run_batch` may be
      overridden when the model decodes several inputs faster together.

    A system with audio fields also sets :attr:`sample_rate`, the rate they
    are resampled to; a system without audio leaves it alone.

    Two surfaces, one for each side. Callers - a notebook, the command
    line, a demo, the ``infer`` stage - use the entry points and nothing
    else: the one-shot call ``model(...)``, :meth:`stream` and
    :meth:`batch`. They bind arguments to the declared fields, convert and
    check every value (a file name becomes :class:`Audio` at
    :attr:`sample_rate`, a list means a batch), and check what comes back.
    Authors implement the hooks :meth:`run`, :meth:`run_stream` and
    :meth:`run_batch`, which therefore see :class:`Audio` at
    :attr:`sample_rate` for every audio field, ``str`` for every text
    field, and never a required field missing. Calling a hook directly
    skips those checks, which is why the hooks are not the API.

    Notes:
        The ``infer`` stage's ``InferenceRunner`` calls the same entry
        points: ``model(**fields)`` for one dataset item, ``model.batch``
        for several, picking the declared inputs out of each item and
        writing each declared output by its kind. An instance serves the
        recipe's ``infer`` stage as it is, with no ``input_key`` and no
        ``output_fn``.

    Examples:
        A system whose model needs the whole input::

            class Inference(InferenceAPI):
                inputs = (Field("speech", "audio"),)
                outputs = (Field("text", "text"),)

                @classmethod
                def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
                    return cls(load_model(locate_pack(tag_or_dir), device=device))

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
    def sample_rate(self) -> Optional[int]:
        """The rate every audio input is resampled to before a hook sees it.

        Only the ``audio`` kind reads this, so only a system with audio
        fields sets it; a text-only system leaves the default. The default,
        ``None``, means the model takes audio at whatever rate it comes -
        right for an enhancement model that adapts to its input, or a test
        set mixing rates as the URGENT challenge does. Each :class:`Audio`
        then keeps its own rate for the hook to read, a bare array is
        refused for carrying none, and an audio output must be returned as
        an :class:`Audio`, since nothing else says its rate.

        Examples:
            >>> class Inference(InferenceAPI):
            ...     sample_rate = 16000          # a fixed-rate frontend
            >>> class Inference(InferenceAPI):
            ...     @property
            ...     def sample_rate(self):       # read off the model
            ...         return self.backend.fs
        """
        return None

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
        list of outputs, as from :meth:`batch`. A value of ``None`` counts
        as not given: an optional field is then left out of the hook's
        arguments, a required one raises ``TypeError``.

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

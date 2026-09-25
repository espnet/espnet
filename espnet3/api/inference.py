"""The inference contract: what a trained ESPnet3 model promises a caller.

Every system under ``espnet3/systems/<name>/`` ships an ``inference.py``
whose ``Inference`` class subclasses :class:`InferenceAPI`. The class states
the fields it takes and returns, and implements :meth:`InferenceAPI.run` for
one sample; the base class does the rest - reading a file, accepting what
Gradio hands over, resampling to the model's rate, checking the declared
fields - so the command line, the MCP server, a Space and a notebook can
drive any system the same way::

    >>> from espnet3.api.inference import load
    >>> model = load("espnet/some_pack")          # meta.yaml names the system
    >>> model("utt.wav")["text"]
    >>> model(speech=(16000, samples))["text"]    # what gr.Audio returns

A system says nothing about *what task* it performs; it says what goes in
and what comes out. A front end that offers ``transcribe`` looks for a
model whose first input is audio and whose outputs hold ``text``, and a
model that answers a conversation declares one ``messages`` field and no
list of the tasks a prompt might ask of it.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import yaml

# What a field can hold. ``audio`` is an :class:`Audio`; ``text`` a ``str``;
# ``segments`` a list of ``{"text", "start", "end", "score"}`` dicts with
# times in seconds, the shape ``espnet align`` prints.
KINDS = frozenset({"audio", "text", "segments"})


@dataclass(frozen=True)
class Field:
    """One input or output of a system.

    Args:
        name: The keyword it is passed or returned as.
        kind: One of :data:`KINDS`; decides how a value is checked and, for a
            front end, which widget or argument type shows it.
        label: What a page calls it. Defaults to the name, spaced out.
        optional: An input the caller may leave out. Outputs are never
            optional.
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
    :meth:`coerce`, resampled to the rate the model wants.
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
        """Read the file at ``path``, at its own rate or resampled to ``rate``."""
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
    def coerce(cls, value: Any, rate: int) -> "Audio":
        """Turn ``value`` into an :class:`Audio` at ``rate``.

        A path is read; a ``(rate, samples)`` pair is what Gradio's audio
        widget returns; an array or tensor without a rate is taken to be at
        ``rate`` already, because there is nothing else it could be.
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


def check_contract(cls: type) -> None:
    """Raise ``TypeError`` unless ``cls`` declares usable inputs and outputs.

    Run on every concrete subclass as it is defined, so a system that gets
    the declaration wrong fails at import, not in a Space at runtime.
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


class InferenceAPI(ABC):
    """What a system implements to be callable from every front end.

    A subclass declares two class attributes and three members:

    - ``inputs`` / ``outputs``: tuples of :class:`Field`; required inputs
      first, so a call by position means one thing.
    - :meth:`from_pretrained`: build one from a packed model directory or a
      Hub tag.
    - :attr:`sample_rate`: the rate the model takes audio at.
    - :meth:`run`: one sample in, one result out, both keyed by field name.

    Calling the instance does the checking and conversion; :meth:`run` sees
    :class:`Audio` at :attr:`sample_rate` for every audio field, ``str`` for
    every text field, and never a missing required one.
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
        """Load a published model: a ``pack_model`` directory or a Hub tag."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The rate audio is resampled to before :meth:`run` sees it."""

    @abstractmethod
    def run(self, **inputs: Any) -> Mapping[str, Any]:
        """Infer one sample; keys in and out are the declared field names."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Bind, check and convert the inputs, run, and check the outputs.

        Positional values fill ``inputs`` in order, so ``model(audio)`` is
        ``model(speech=audio)`` for any transcriber.
        """
        bound = self._bind(args, kwargs)
        result = self.run(**bound)
        if not isinstance(result, Mapping):
            raise TypeError(
                f"{type(self).__qualname__}.run returned "
                f"{type(result).__name__}, not a mapping"
            )
        out: dict[str, Any] = {}
        for f in self.outputs:
            if f.name not in result:
                raise RuntimeError(
                    f"{type(self).__qualname__}.run did not return {f.name!r}"
                )
            out[f.name] = self._check(f, result[f.name], output=True)
        for name, value in result.items():
            out.setdefault(name, value)
        return out

    def _bind(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
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
        bound: dict[str, Any] = {}
        for f in self.inputs:
            value = values.get(f.name)
            if value is None:
                if f.optional:
                    continue
                raise TypeError(f"{type(self).__qualname__} needs {f.name!r}")
            bound[f.name] = self._check(f, value, output=False)
        return bound

    def _check(self, f: Field, value: Any, *, output: bool) -> Any:
        where = "returned" if output else "given"
        if f.kind == "audio":
            if output and isinstance(value, np.ndarray):
                # a model's own output is at its own rate
                return Audio(value, self.sample_rate)
            return Audio.coerce(value, self.sample_rate)
        if f.kind == "text":
            if not isinstance(value, str):
                raise TypeError(
                    f"{f.name!r} {where} as {type(value).__name__}, must be str"
                )
            return value
        if f.kind == "segments":
            if not isinstance(value, list) or not all(
                isinstance(s, Mapping) and {"text", "start", "end"} <= set(s)
                for s in value
            ):
                raise TypeError(
                    f"{f.name!r} {where} must be a list of dicts "
                    "with text, start and end"
                )
            return value
        raise AssertionError(f.kind)  # KINDS and _check disagree


# A system renamed after bundles were published under its old name: old name
# to current directory. ``load`` looks the name in ``meta.yaml`` up here, so a
# rename is one row and every bundle already on the Hub keeps loading.
SYSTEM_ALIASES: dict[str, str] = {}


def locate_pack(tag_or_dir: str | Path) -> Path:
    """Return the directory of a ``pack_model`` bundle, downloading a Hub tag."""
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

    The bundle's ``meta.yaml`` names the system that trained it; that
    system's ``inference.Inference`` loads it. ``system`` overrides the name,
    for bundles packed before it was recorded and for models that are not
    ESPnet3 bundles at all. A name a system has since given up is followed
    through :data:`SYSTEM_ALIASES`.
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

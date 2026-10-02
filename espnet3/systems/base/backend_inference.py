"""An ``Inference`` over one backend object, with the boilerplate done once.

Most systems wrap a single object that does the work - an ESPnet2
``Speech2Text``, a ``Text2Speech``, a ``SeparateSpeech`` - and would
otherwise each repeat the same three things: build that object from its
own arguments or from a published bundle, find the rate it works at, and
hand it to ``run``. :class:`BackendInference` does those, so a system's
``inference.py`` is its declaration and ``run``::

    class Inference(BackendInference):
        backend_class = "espnet2.bin.asr_inference.Speech2Text"
        inputs = (Field("speech", "audio"),)
        outputs = (Field("text", "text"),)

        def run(self, speech):
            return {"text": self.backend(speech.array)[0][0]}

That class is built three ways, all ending in ``self.backend``:

- ``Inference.from_pretrained(tag_or_dir)``: the bundle's own
  ``conf/inference.yaml`` builds the backend, through the bundle's provider
  and without importing bundled code.
- ``Inference(asr_train_config=..., asr_model_file=...)``: the backend's own
  arguments, which is how ``inference.yaml``'s ``model`` names the class
  for the ``infer`` stage (the provider adds ``device``).
- ``Inference(backend)``: one already built.

A system whose model is not one object - a SpeechLM behind a server, a
pipeline of several - subclasses :class:`InferenceAPI` directly instead.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar, Optional

import humanfriendly

from espnet3.api.inference import InferenceAPI, locate_pack
from espnet3.publication.inference_model import load_backend

DEFAULT_RATE = 16000


def _resolve(dotted: str) -> type:
    """Import ``package.module.Class`` and return the class."""
    module_name, _, class_name = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _rate(value: Any) -> int:
    """Return an int rate from an int or ESPnet2's ``"16k"`` spelling."""
    if isinstance(value, str):
        value = humanfriendly.parse_size(value)
    return int(value)


class BackendInference(InferenceAPI):
    """The base for a system that wraps one backend object.

    A subclass declares ``backend_class``, ``inputs`` and ``outputs`` and
    implements :meth:`InferenceAPI.run` (or ``run_stream``) over
    ``self.backend``; building, loading and the rate are done here.

    Args:
        backend: An already-built backend, or ``None`` to build one.
        device: Where to build the backend when building one.
        backend_class: A dotted class path overriding the class attribute
            for this instance - how a recipe picks a variant such as
            ``espnet2.bin.asr_transducer_inference.Speech2Text``.
        **kwargs: The backend class's own constructor arguments when
            building one.

    Raises:
        TypeError: If ``kwargs`` or ``backend_class`` are given together
            with a built ``backend``, or nothing says which class to build.

    Examples:
        >>> model = Inference(asr_train_config="exp/config.yaml",
        ...                   asr_model_file="exp/valid.acc.ave.pth")
        >>> model("utt.wav")["text"]
        >>> Inference.from_pretrained("espnet/some_pack", device="cuda:0")
        >>> Inference(speech2text)   # built elsewhere

        In ``inference.yaml``, where the class takes the place of the
        backend and keeps its arguments::

            model:
              _target_: espnet3.systems.asr.inference.Inference
              asr_train_config: ${exp_dir}/config.yaml
              asr_model_file: ${exp_dir}/valid.acc.ave.pth
              beam_size: 10
    """

    backend_class: ClassVar[Optional[str]] = None

    def __init__(
        self,
        backend: Any = None,
        *,
        device: str = "cpu",
        backend_class: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Keep a built backend, or build one from its own arguments."""
        if backend is None:
            path = backend_class or type(self).backend_class
            if not path:
                raise TypeError(
                    f"{type(self).__qualname__} declares no backend_class; "
                    "pass a built backend or set backend_class"
                )
            backend = _resolve(path)(device=device, **kwargs)
        elif kwargs or backend_class:
            raise TypeError(
                f"arguments {sorted(kwargs)} given for building a backend, "
                "but a built one was too"
            )
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls, tag_or_dir: str | Path, *, device: str = "cpu", **kwargs: Any
    ) -> "BackendInference":
        """Load a ``pack_model`` bundle by directory or Hub tag.

        The bundle's ``conf/inference.yaml`` says how to build the model;
        it is built through the bundle's provider on ``device``, without
        importing the recipe's code: the outputs are fixed by the
        declaration, so the recipe's ``output_fn`` is never needed. A
        recipe that names this class as its ``model`` builds the
        ``Inference`` itself, which is returned as it is; an older one
        names the backend, which is wrapped.

        Args:
            tag_or_dir: A ``pack_model`` output directory, or a Hub tag.
            device: Where to build the backend.
            **kwargs: None are taken; a bundle is complete as packed.

        Raises:
            TypeError: If ``kwargs`` are given, or the bundle builds an
                ``Inference`` of another class.
            ValueError: If the bundle's model needs the bundle's own code.

        Examples:
            >>> Inference.from_pretrained("exp/train/model_pack")
            >>> Inference.from_pretrained("espnet/some_pack", device="cuda:0")
        """
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        built = load_backend(locate_pack(tag_or_dir), device=device)
        if isinstance(built, InferenceAPI):
            # the bundle's inference.yaml names the Inference itself, as a
            # recipe on the APIRunner does: it is the model, not a backend
            if not isinstance(built, cls):
                raise TypeError(
                    f"the bundle builds {type(built).__name__}, not {cls.__name__}"
                )
            return built
        return cls(built)

    @property
    def sample_rate(self) -> Optional[int]:
        """The rate the backend works at, read off the backend.

        In order: a ``sample_rate`` or ``fs`` attribute; the ESPnet2
        training config's ``frontend_conf.fs`` (an int or ``"16k"``) on
        any ``*_train_args`` the backend keeps; else 16 kHz. Override when
        the backend says it some other way, and return ``None`` for a
        backend that takes any rate (``SeparateSpeech`` takes ``fs`` per
        call, so an enhancement system passes ``speech.rate`` through).
        """
        backend = self.backend
        for name in ("sample_rate", "fs"):
            rate = getattr(backend, name, None)
            if rate:
                return _rate(rate)
        for name, value in (
            list(vars(backend).items()) if hasattr(backend, "__dict__") else []
        ):
            if name.endswith("_train_args"):
                conf = getattr(value, "frontend_conf", None) or {}
                if conf.get("fs"):
                    return _rate(conf["fs"])
        return DEFAULT_RATE

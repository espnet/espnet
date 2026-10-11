"""An ``Inference`` over one backend object, with the boilerplate done once.

Many systems wrap a single object that does the work - an ESPnet2
``Speech2Text``, a ``Text2Speech``, a ``SeparateSpeech``, an F5-TTS or
kNN-VC model - and would otherwise each repeat the same things: build that
object from its own arguments or from a published bundle, keep it for
``run``, and say the rate it works at. Nothing here is specific to one
toolkit: what a backend's config looks like is the system's knowledge
(the ESPnet2 ASR system reads ``frontend_conf``), not this class's.
:class:`BackendInference` does those, so a system's
``inference.py`` is its declaration and ``run``::

    class Inference(BackendInference):
        backend_class = "espnet2.bin.asr_inference.Speech2Text"
        inputs = (Field("speech", "audio"),)
        outputs = (Field("text", "text"),)

        def run(self, speech):
            return {"text": self.backend(speech.array)[0][0]}

That class is built three ways, all ending in ``self.backend``:

- ``Inference.from_pretrained(tag_or_dir)``: :func:`load_model` builds
  the backend from the bundle's ``conf/inference.yaml`` ``model``, without
  any provider and without importing bundled code. A Hub tag that is no
  bundle but a model the backend's own toolkit published (an ESPnet2
  ``Speech2Text`` on the Hub) goes to the backend class's own
  ``from_pretrained``.
- ``Inference(asr_train_config=..., asr_model_file=...)``: the backend's own
  arguments, which is how ``inference.yaml``'s ``model`` names the class
  for the ``infer`` stage (the provider adds ``device``).
- ``Inference(backend)``: one already built.

A hook receives audio channels-first, as :class:`Audio` keeps it:
``(samples,)`` for one channel and ``(channels, samples)`` for several. An
ESPnet2 backend takes several channels the other way round, as
``soundfile`` reads them, so a system passing several channels to one
hands it ``speech.array.T``; the ESPnet2 ASR system does this for a model
whose frontend uses every channel.

A system whose model is not one object - a SpeechLM behind a server, a
pipeline of several - subclasses :class:`InferenceAPI` directly instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

import humanfriendly
from hydra.utils import get_class

from espnet3.api.inference import InferenceAPI, ModelTagError, load_model, locate_pack
from espnet3.api.inference.loading import _disallowed_targets


def parse_rate(value: Any) -> int:
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
              _target_: espnet3.systems.esp2_asr.inference.Inference
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
            backend = self._backend_type(backend_class)(device=device, **kwargs)
        elif kwargs or backend_class:
            raise TypeError(
                f"arguments {sorted(kwargs)} given for building a backend, "
                "but a built one was too"
            )
        self.backend = backend

    @classmethod
    def _backend_type(cls, backend_class: Optional[str] = None) -> type:
        """Return the backend class to build: the argument, else the declared one.

        The declared ``backend_class`` is the system's own code. One passed
        as an argument can come from a published bundle's
        ``conf/inference.yaml``, so it must be an ESPnet2 or ESPnet3 class,
        checked as a bundle's ``_target_`` is - by name before anything is
        imported, then by what it resolves to - since importing it runs
        that module and building it runs its constructor.

        Raises:
            TypeError: If neither names a class.
            ValueError: If the argument names anything but an ESPnet class.
        """
        if backend_class:
            disallowed = _disallowed_targets({"_target_": backend_class})
            if disallowed:
                raise ValueError(
                    "backend_class must be an ESPnet2 or ESPnet3 class; "
                    f"{', '.join(disallowed)} is not"
                )
        path = backend_class or cls.backend_class
        if not path:
            raise TypeError(
                f"{cls.__qualname__} declares no backend_class; "
                "pass a built backend or set backend_class"
            )
        return get_class(path)

    @classmethod
    def from_pretrained(
        cls, tag_or_dir: str | Path, *, device: str = "cpu", **kwargs: Any
    ) -> "BackendInference":
        """Load a ``pack_model`` bundle, or a model the backend publishes.

        The bundle's ``conf/inference.yaml`` says how to build the model;
        :func:`load_model` builds its ``model`` on ``device``, without any
        provider and without importing the recipe's code: the outputs are fixed by the
        declaration, so the recipe's ``output_fn`` is never needed. A
        recipe that names this class as its ``model`` builds the
        ``Inference`` itself, which is returned as it is; an older one
        names the backend, which is wrapped.

        Args:
            tag_or_dir: A ``pack_model`` output directory, or a Hub tag.
            device: Where to build the backend.
            **kwargs: Any argument the packed model's constructor takes,
                replacing the packed value, as ESPnet2's ``from_pretrained``
                takes them. Nothing is singled out: for an ESPnet2 ASR
                bundle that is every ``Speech2Text`` argument - the
                decoding ones (``beam_size``, ``ctc_weight``,
                ``lm_weight``, ``penalty``, ``nbest``, ``maxlenratio``,
                ...) and the rest (``lm_file``, ``dtype``, ...). An
                argument the model does not take is a ``TypeError``
                naming it.

        A tag that is not a ``pack_model`` bundle - a model the backend's
        own toolkit published, such as an ESPnet2 ``Speech2Text`` on the
        Hub - goes to the backend class's own ``from_pretrained`` when it
        has one, with the same ``device`` and overrides, and the backend it
        returns is wrapped. That is the whole of it: which tags a backend
        can read is the backend's to say, not this class's. Only the
        declared ``backend_class`` reads such a tag: a ``backend_class``
        override is for a bundle's ``conf/inference.yaml``, and is refused
        here.

        Raises:
            ModelTagError: If the tag is neither a ``pack_model`` bundle nor
                a model the backend class reads, or the bundle builds an
                ``Inference`` of another class.
            ValueError: If the bundle's model needs the bundle's own code.
            TypeError: If ``backend_class`` is given for a tag that is no
                bundle.

        Examples:
            >>> Inference.from_pretrained("exp/train/model_pack")
            >>> Inference.from_pretrained("espnet/some_pack", device="cuda:0")
            >>> Inference.from_pretrained(
            ...     "espnet/some_pack", beam_size=5, ctc_weight=0.3, nbest=3
            ... )
            >>> Inference.from_pretrained("espnet/an_espnet2_asr_model")
        """
        try:
            pack = locate_pack(tag_or_dir)
        except ModelTagError:
            # the download is in the cache now; the backend reads it from there
            backend = cls._published_backend(str(tag_or_dir), device, kwargs)
            if backend is None:
                raise
            return cls(backend)
        built = load_model(pack, device=device, overrides=kwargs)
        if isinstance(built, InferenceAPI):
            # the bundle's inference.yaml names the Inference itself, as a
            # recipe names its Inference as the model: it is the model, not a backend
            if not isinstance(built, cls):
                raise ModelTagError(
                    f"the bundle builds {type(built).__name__}, not {cls.__name__}"
                )
            return built
        return cls(built)

    @classmethod
    def _published_backend(
        cls, tag: str, device: str, overrides: dict[str, Any]
    ) -> Any:
        """Read a tag the backend class publishes itself, or ``None`` if it cannot.

        Goes through ESPnet2's ``build_pretrained``, so a tag for another
        kind of model is the same ``ModelTagError`` the command line and
        ``espnet.load`` give. The declared backend class does the reading;
        a ``backend_class`` override among ``overrides`` is refused rather
        than passed on to a constructor that does not take it.

        Raises:
            TypeError: If ``overrides`` names a ``backend_class``.
        """
        backend_type = cls._backend_type() if cls.backend_class else None
        if not callable(getattr(backend_type, "from_pretrained", None)):
            return None
        if "backend_class" in overrides:
            raise TypeError(
                f"backend_class={overrides['backend_class']!r}: a tag the backend "
                f"publishes is read by the declared {cls.backend_class}; another "
                "backend is picked in a pack_model bundle's conf/inference.yaml"
            )
        from espnet2.utils.pretrained import build_pretrained

        return build_pretrained(
            backend_type,
            tag,
            device,
            f"{cls.__module__}.{cls.__qualname__}.from_pretrained",
            "Pass a model this system serves, or a pack_model bundle.",
            **overrides,
        )

    @property
    def sample_rate(self) -> Optional[int]:
        """The rate the backend works at, read off the backend.

        A ``sample_rate`` or ``fs`` attribute on the backend, when it has
        one. There is no default and no knowledge of any toolkit's config
        here: a guessed rate would be silently wrong for a model at
        another, so a system whose backend says it some other way
        overrides this (the ESPnet2 ASR system reads its frontend config),
        and one whose backend takes any rate returns ``None``.

        Raises:
            TypeError: If the backend has neither attribute.
        """
        backend = self.backend
        for name in ("sample_rate", "fs"):
            rate = getattr(backend, name, None)
            if rate:
                return parse_rate(rate)
        raise TypeError(
            f"{type(self).__qualname__} cannot tell the rate from its backend "
            f"({type(backend).__name__}); override sample_rate, or return None "
            "for a backend that takes any rate"
        )

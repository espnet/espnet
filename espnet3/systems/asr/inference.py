"""The ESPnet2 ASR system's side of :mod:`espnet3.api.inference`.

The model behind an ``asr`` bundle is
:class:`espnet2.bin.asr_inference.Speech2Text`, which takes one waveform and
returns an n-best list of ``(text, tokens, token_ids, hypothesis)``. This
adapter is the whole distance between that and the contract: the best
text, under the key every transcriber uses, from audio at the rate the
frontend was trained at.

Examples:
    From a bundle, by directory or Hub tag::

        >>> from espnet3.systems.asr.inference import Inference
        >>> model = Inference.from_pretrained("exp/train/model_pack")
        >>> model("utt.wav")
        {'text': 'hello world'}

    Or, without naming the system, through the bundle's ``meta.yaml``::

        >>> from espnet3.api.inference import load
        >>> load("espnet/some_asr_pack", device="cuda:0")("utt.wav")["text"]

    In the ``infer`` stage, as the model the runner calls, with the
    dataset's ``speech`` field as the input and ``text.scp`` written from
    the result. ``inference.yaml`` names this class where it named
    ``Speech2Text``, with the same keys, and the provider builds it on
    the device it picks::

        model:
          _target_: espnet3.systems.asr.inference.Inference
          asr_train_config: ${exp_dir}/config.yaml
          asr_model_file: ${exp_dir}/valid.acc.ave.pth
          beam_size: 10
        input_key: speech
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import humanfriendly

from espnet3.api.inference import Audio, Field, InferenceAPI, locate_pack
from espnet3.publication.inference_model import load_backend


class Inference(InferenceAPI):
    """Transcribe with an ESPnet2 ASR model.

    Args:
        speech2text: A built :class:`espnet2.bin.asr_inference.Speech2Text`,
            or anything called the same way. When omitted, one is built
            from ``device`` and ``kwargs``.
        device: Where to build the ``Speech2Text`` when building one.
        **kwargs: ``Speech2Text``'s own arguments - ``asr_train_config``,
            ``asr_model_file``, ``beam_size`` and the rest - when building
            one; none are allowed together with ``speech2text``.

    Raises:
        TypeError: If ``kwargs`` are given along with ``speech2text``.

    Examples:
        >>> model = Inference(asr_train_config="exp/config.yaml",
        ...                   asr_model_file="exp/valid.acc.ave.pth")
        >>> model(np.zeros(16000, dtype=np.float32))
        {'text': ''}
        >>> model.sample_rate
        16000
        >>> Inference(speech2text)          # one already built
    """

    inputs = (Field("speech", "audio", "Speech"),)
    outputs = (Field("text", "text", "Transcription"),)

    def __init__(
        self, speech2text: Any = None, *, device: str = "cpu", **kwargs: Any
    ) -> None:
        """Wrap a built ``Speech2Text``, or build one from its arguments."""
        if speech2text is None:
            from espnet2.bin.asr_inference import Speech2Text

            speech2text = Speech2Text(device=device, **kwargs)
        elif kwargs:
            raise TypeError(
                f"Speech2Text arguments {sorted(kwargs)} given, but a "
                "speech2text was too"
            )
        self.speech2text = speech2text

    @classmethod
    def from_pretrained(
        cls, tag_or_dir: str | Path, *, device: str = "cpu", **kwargs: Any
    ) -> "Inference":
        """Load a ``pack_model`` bundle by directory or Hub tag.

        The bundle's ``conf/inference.yaml`` says how to build the
        ``Speech2Text``; it is built through the bundle's provider, on
        ``device``, without importing the recipe's own code: the output is
        fixed here, so the recipe's ``output_fn`` is never needed.

        Args:
            tag_or_dir: A ``pack_model`` output directory, or a Hub tag.
            device: Where to build the model, ``"cpu"`` or ``"cuda:0"``.
            **kwargs: None are taken; an ASR bundle is complete as packed.

        Returns:
            A ready instance.

        Raises:
            TypeError: If any ``kwargs`` are given.
            ValueError: If the bundle's model itself needs the bundle's
                code to build (see
                :func:`espnet3.publication.inference_model.load_backend`).

        Examples:
            >>> Inference.from_pretrained("exp/train/model_pack")
            >>> Inference.from_pretrained("espnet/some_asr_pack", device="cuda:0")
        """
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        return cls(load_backend(locate_pack(tag_or_dir), device=device))

    @property
    def sample_rate(self) -> int:
        """The rate the frontend was trained at, from the packed config.

        Read from ``asr_train_args.frontend_conf.fs``, which ESPnet2 writes
        as an int or as ``"16k"``; 16 kHz when the config says nothing.
        """
        args = getattr(self.speech2text, "asr_train_args", None)
        conf = getattr(args, "frontend_conf", None) or {}
        rate = conf.get("fs", 16000)
        if isinstance(rate, str):  # "16k", as ESPnet2 configs write it
            rate = humanfriendly.parse_size(rate)
        return int(rate)

    def run(self, speech: Audio) -> Mapping[str, Any]:
        """Return the best hypothesis' text.

        Args:
            speech: The utterance, at :attr:`sample_rate`.

        Returns:
            ``{"text": str}``. For a joint enhancement-and-ASR model, which
            returns one n-best list per speaker, the first speaker's.
        """
        nbest = self.speech2text(speech.array)
        best = nbest[0]
        # Speech2Text returns (text, tokens, token_ids, hyp); a joint enh+ASR
        # model returns one such list per speaker, of which this is the first.
        if isinstance(best, list):
            best = best[0]
        return {"text": best[0]}

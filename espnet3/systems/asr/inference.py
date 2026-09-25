"""The ESPnet2 ASR system's side of :mod:`espnet3.api.inference`.

The model behind an ``asr`` bundle is :class:`espnet2.bin.asr_inference.Speech2Text`,
which takes one waveform and returns an n-best list of
``(text, tokens, token_ids, hypothesis)``. This adapter is the whole
distance between that and the contract: the best text, under the key every
transcriber uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import humanfriendly

from espnet3.api.inference import Audio, Field, InferenceAPI
from espnet3.publication.inference_model import InferenceModel


class Inference(InferenceAPI):
    """Transcribe with a packed ESPnet2 ASR model."""

    task = "transcribe"
    inputs = (Field("speech", "audio", "Speech"),)
    outputs = (Field("text", "text", "Transcription"),)

    def __init__(self, speech2text: Any) -> None:
        """Wrap a built ``Speech2Text``; :meth:`from_pretrained` builds one."""
        self.speech2text = speech2text

    @classmethod
    def from_pretrained(
        cls, tag_or_dir: str | Path, *, device: str = "cpu", **kwargs: Any
    ) -> "Inference":
        """Load a ``pack_model`` bundle by directory or Hub tag.

        The bundle's ``conf/inference.yaml`` says how to build the
        ``Speech2Text``; ``kwargs`` are not used, because an ASR bundle is
        complete as packed. Bundled recipe code is never imported: the
        contract's output is fixed here, so the recipe's ``output_fn`` is
        not needed.
        """
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        if Path(tag_or_dir).is_dir():
            model = InferenceModel.from_packed(tag_or_dir, device=device)
        else:
            model = InferenceModel.from_pretrained(str(tag_or_dir), device=device)
        return cls(model.model)

    @property
    def sample_rate(self) -> int:
        """The rate the frontend was trained at, from the packed config."""
        args = getattr(self.speech2text, "asr_train_args", None)
        conf = getattr(args, "frontend_conf", None) or {}
        rate = conf.get("fs", 16000)
        if isinstance(rate, str):  # "16k", as ESPnet2 configs write it
            rate = humanfriendly.parse_size(rate)
        return int(rate)

    def run(self, speech: Audio) -> Mapping[str, Any]:
        """Return the best hypothesis' text."""
        nbest = self.speech2text(speech.array)
        best = nbest[0]
        # Speech2Text returns (text, tokens, token_ids, hyp); a joint enh+ASR
        # model returns one such list per speaker, of which this is the first.
        if isinstance(best, list):
            best = best[0]
        return {"text": best[0]}

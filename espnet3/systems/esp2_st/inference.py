"""The ESPnet2 ST system's side of :mod:`espnet3.api.inference`.

The model behind an ``esp2_st`` bundle is
:class:`espnet2.bin.st_inference.Speech2Text`, which takes one waveform and
returns an n-best list of ``(text, tokens, token_ids, hypothesis)``. This
adapter is the whole distance between that and the contract: the best
translation, under the key every system that produces text uses. Building
the ``Speech2Text``, loading a bundle and the frontend's rate come from
:class:`espnet3.systems.base.backend_inference.BackendInference`.

Examples:
    From a bundle, by directory or Hub tag::

        >>> from espnet3.systems.esp2_st.inference import Inference
        >>> model = Inference.from_pretrained("exp/train/model_pack")
        >>> model("utt.wav")
        {'text': 'hallo welt'}

    Or, without naming the system, through the bundle's ``meta.yaml``::

        >>> from espnet3.api.inference import load
        >>> load("espnet/some_st_pack", device="cuda:0")("utt.wav")["text"]

    In the ``infer`` stage, ``inference.yaml`` names this class where it
    named ``Speech2Text``, with the same arguments::

        model:
          _target_: espnet3.systems.esp2_st.inference.Inference
          st_train_config: ${exp_dir}/config.yaml
          st_model_file: ${exp_dir}/valid.acc.ave_10best.pth
          beam_size: 10
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)
from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference, parse_rate

logger = logging.getLogger(__name__)

# espnet2.asr.frontend.default.DefaultFrontend(fs=16000): what an ESPnet2
# model runs at when its training config says nothing about the rate.
DEFAULT_FRONTEND_RATE = 16000

# What egs2 scores an unsubsamplable utterance as (st.sh writes a blank line
# rather than dropping the row), kept so the two agree.
TOO_SHORT_TEXT = " "


class Inference(BackendInference):
    """Translate speech with an ESPnet2 ST model.

    Built three ways, all ending in ``self.backend`` (see
    :class:`BackendInference`):

    - ``Inference.from_pretrained(tag_or_dir, device=...)`` from a
      ``pack_model`` bundle or Hub tag;
    - ``Inference(st_train_config=..., st_model_file=..., beam_size=...)``
      from ``Speech2Text``'s own arguments, which is what ``inference.yaml``
      does;
    - ``Inference(speech2text)`` around one already built.

    Called with one utterance (a path, a ``(rate, samples)`` pair, an array
    or an :class:`~espnet3.api.inference.Audio`) it returns
    ``{"text": str}``. ``espnet2.bin.st_inference.Speech2Text`` decodes one
    utterance per call, so ``model.batch(items)`` is one call each.

    Examples:
        >>> model = Inference.from_pretrained("espnet/some_st_pack")
        >>> model("utt.wav")
        {'text': 'hallo welt'}
        >>> model.sample_rate       # the frontend's rate, read off the config
        16000
    """

    backend_class = "espnet2.bin.st_inference.Speech2Text"
    inputs = (Field("speech", "audio", "Speech"),)
    outputs = (Field("text", "text", "Translation"),)

    @property
    def sample_rate(self) -> int:
        """The frontend's rate: ``frontend_conf.fs``, else ESPnet2's 16 kHz.

        An ESPnet2 training config may leave ``frontend_conf`` empty, as the
        mini_an4 recipe does; the frontend it builds then runs at
        :class:`~espnet2.asr.frontend.default.DefaultFrontend`'s own default
        of 16 kHz, so that is the rate, not a guess. A backend that is not a
        ``Speech2Text`` (no ``st_train_args``) falls back to the base class,
        which raises.
        """
        args = getattr(self.backend, "st_train_args", None)
        if args is None:
            return super().sample_rate  # not a Speech2Text: its own, or an error
        conf = getattr(args, "frontend_conf", None) or {}
        return parse_rate(conf.get("fs", DEFAULT_FRONTEND_RATE))

    def run(self, speech: Audio) -> Mapping[str, Any]:
        """Return the best hypothesis' translation.

        Args:
            speech: The utterance, at :attr:`sample_rate`.

        Returns:
            ``{"text": str}``. An utterance too short for the encoder to
            subsample gets :data:`TOO_SHORT_TEXT` instead of raising: egs2
            does not apply its duration filter to test sets (``st.sh``:
            "NOT applying to test_sets to keep original data"), so a test
            set legitimately holds utterances below the few frames a
            conv2d-subsampling encoder needs - 54 of tst-COMMON's 2,641,
            the shortest 0.050 s. espnet2 handles this in its CLI loop
            rather than in ``Speech2Text.__call__``, which is what espnet3
            calls, so one such utterance would otherwise abort the set.
            Scoring it as a miss is what egs2 does; dropping it would
            inflate BLEU by taking the hardest items out of the
            denominator.
        """
        try:
            results = self.backend(speech.mono().array)
        except TooShortUttError as e:
            logger.warning("%s: scoring an empty translation", e)
            return {"text": TOO_SHORT_TEXT}
        return {"text": _best_text(results)}


def _best_text(results: Any) -> str:
    """Return the best hypothesis' text from what ``Speech2Text`` returned.

    A multi-decoder model returns ``(translations, transcripts)`` rather
    than one n-best list; only the translation is this system's output.
    """
    if isinstance(results, tuple):
        results = results[0]
    return results[0][0] or ""

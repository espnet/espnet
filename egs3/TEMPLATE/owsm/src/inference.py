"""Decoding helpers for OWSM, whose test sets mix tasks.

``conf/inference.yaml`` uses both halves::

    model:
      _target_: src.inference.Speech2TextOWSM
    input_key: [speech, text]
    output_fn: src.inference.build_output

A recipe re-exports this module as its own ``src/inference.py``; ``path.sh``
puts the recipe on the path, so ``src.`` resolves there.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)

logger = logging.getLogger(__name__)

# The leading tags of a reference: <lang><task>, e.g. <eng><asr> or
# <eng><st_deu>.
_PREFIX = re.compile(r"^\s*<(?P<lang>[^<>]+)>\s*<(?P<task>[^<>]+)>")


def prompt_of(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(lang_sym, task_sym)`` from a reference's leading tags.

    OWSM is told which language and task to produce; egs2 does this per test
    set with ``--lang_sym``/``--task_sym``, but one OWSM cache holds every
    direction, so it has to be per utterance instead.

    This gives the model the reference's *tags*, never its words, so the
    transcript or translation is still produced from audio alone. It does mean
    language and task identification are not being evaluated -- the same
    limitation egs2 has, where both are fixed for the whole set.
    """
    if not text:
        return None, None
    match = _PREFIX.match(text)
    if not match:
        return None, None
    return f"<{match.group('lang')}>", f"<{match.group('task')}>"


class Speech2TextOWSM(Speech2Text):
    """``Speech2Text`` prompted per utterance, tolerant of unsubsamplable audio.

    Two departures from the base class, both forced by decoding a mixture:

    * the language and task come from each utterance rather than from one
      setting for the whole test set;
    * a clip too short to subsample yields an empty hypothesis instead of
      aborting the set. espnet2 handles this in its CLI loop rather than in
      ``__call__``, and espnet3 calls the class directly.
    """

    def __call__(self, speech, text: Optional[str] = None, **kwargs):
        """Decode one utterance.

        Args:
            speech: Audio of shape ``(nsamples,)``.
            text: The reference, read only for its ``<lang><task>`` prefix.
            **kwargs: Forwarded to :class:`Speech2Text`.

        Returns:
            The n-best list of ``(text, token, token_int, text_nospecial,
            hyp)``, or a single empty hypothesis when the audio is too short to
            subsample.
        """
        lang_sym, task_sym = prompt_of(text)
        if lang_sym is not None:
            kwargs.setdefault("lang_sym", lang_sym)
            kwargs.setdefault("task_sym", task_sym)
        try:
            return super().__call__(speech, **kwargs)
        except TooShortUttError:
            # Keep the utterance so its reference still counts against the
            # score; dropping it would flatter the model. The arity has to
            # match Speech2Text's own five-tuple or build_output misreads it.
            logger.warning("utterance too short to subsample; emitting empty text")
            return [("", [], [], "", None)]


def build_output(data: Dict[str, Any], model_output, idx) -> Dict[str, Any]:
    """Turn one decoding result into the row ESPnet3 writes to SCP.

    ``Speech2Text`` returns ``(text, token, token_int, text_nospecial, hyp)``
    per hypothesis. ``hyp`` takes ``text_nospecial``, which is what
    ``egs2/TEMPLATE/s2t1/s2t.sh`` scores -- it drops every ``<...>`` token, so
    a wrong language tag is not also counted as a word error. ``hyp_raw`` keeps
    the tagged text, since the tags are the only record of what the model
    thought it was doing.

    ``ref`` keeps its tags: the metrics read the task off them to decide which
    rows are theirs, and strip them before scoring.
    """
    best = model_output[0]
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": best[3],
        "hyp_raw": best[0],
        "ref": data.get("text", ""),
    }

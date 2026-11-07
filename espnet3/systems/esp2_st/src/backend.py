"""The ESPnet2 ST backend, tolerant of audio too short to subsample."""

from __future__ import annotations

import logging

from espnet2.bin.st_inference import Speech2Text
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)

logger = logging.getLogger(__name__)

# What egs2 scores an unsubsamplable utterance as: st.sh keeps the row and
# scores a blank line rather than dropping it, so the two agree.
TOO_SHORT_TEXT = " "


class Speech2TextShortSafe(Speech2Text):
    """``Speech2Text`` that survives utterances too short to subsample.

    egs2 does not apply the 0.1-20 s filter to test sets (``st.sh``: "NOT
    applying to test_sets to keep original data"), so tst-COMMON contains 54
    of 2,641 utterances below the frames a conv2d-subsampling conformer
    needs -- the shortest is 0.050 s. Decoding one raises
    ``TooShortUttError``.

    espnet2 handles this in its CLI loop (``espnet2/bin/st_inference.py``),
    not inside ``Speech2Text.__call__``, which is what the infer stage
    reaches through the runner. One such utterance would otherwise abort the
    whole test set.

    The fallback mirrors egs2: emit :data:`TOO_SHORT_TEXT` and keep the
    utterance, so its reference is still scored as a miss rather than
    dropped -- dropping would inflate BLEU by taking the hardest items out
    of the denominator.

    It lives in the backend rather than in the runner because
    ``format_output`` runs after the model call, and so cannot catch an
    exception raised during it.
    """

    def __call__(self, *args, **kwargs):
        """Decode one utterance, tolerating one too short to subsample.

        Args:
            *args: Forwarded to :class:`Speech2Text`, whose only argument is
                ``speech``.
            **kwargs: Likewise.

        Returns:
            The n-best list of ``(text, token, token_int, hyp)``, or a single
            :data:`TOO_SHORT_TEXT` hypothesis when the audio is too short to
            subsample.
        """
        try:
            return super().__call__(*args, **kwargs)
        except TooShortUttError as e:
            logger.warning("%s: scoring a blank translation", e)
            hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
            return [(TOO_SHORT_TEXT, ["<space>"], [2], hyp)]

"""CER, WER and TER, scored the way egs2's s2t.sh scores them.

``egs2/TEMPLATE/s2t1/s2t.sh`` stage 13 is the whole of OWSM's evaluation: it
tokenizes to char, word and BPE and runs sclite over each, giving ``cer``,
``wer`` and ``ter``. There is no BLEU anywhere in any owsm recipe, and no
task-aware scoring -- translation rows are scored against their translations as
if they were transcripts. These classes reproduce that, with jiwer in place of
sclite.

* Tags are removed from both sides, using the recipe's own ``nlsyms``
inventory so that only symbols the data can actually contain are stripped.

* No text cleaner by default. ``ref_cleaner`` and ``hyp_cleaner`` are separate
in s2t.sh and both default to ``none``. ``egs2/owsm_v3/s2t1/run.sh:18`` carries
``# --cleaner whisper_en --hyp_cleaner whisper_en`` commented out; pass
``ref_cleaner: [whisper_en]`` to turn it on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.owsm.metrics.tags import strip_tags, tag_pattern
from espnet3.systems.owsm.symbols import load_symbols

logger = logging.getLogger(__name__)


class ErrorRate(BaseMetric):
    """Base for the three rates s2t.sh reports; subclasses pick the unit."""

    #: Key under which the score is reported.
    KEY = ""

    def __init__(
        self,
        nlsyms: Any = None,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        ref_cleaner: Iterable[str] | None = None,
        hyp_cleaner: Iterable[str] | None = None,
        remove_tags: bool = True,
    ) -> None:
        """Initialize the metric.

        Args:
            nlsyms: The recipe's OWSM symbols -- the same hydra callable, file
                or list that ``tokenizer.nlsyms`` names. Required unless
                ``remove_tags`` is False.
            ref_key: Key name for reference entries.
            hyp_key: Key name for hypothesis entries.
            ref_cleaner: TextCleaner types for the reference, as s2t.sh's
                ``--cleaner``.
            hyp_cleaner: TextCleaner types for the hypothesis, as s2t.sh's
                ``--hyp_cleaner``. Separate because s2t.sh keeps them separate.
            remove_tags: Strip OWSM tags from both sides, so they are
                comparable. On by default; False reproduces s2t.sh, whose own
                removal does not work on OWSM text.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.ref_cleaner = TextCleaner(ref_cleaner)
        self.hyp_cleaner = TextCleaner(hyp_cleaner)
        self.remove_tags = remove_tags
        if remove_tags and nlsyms is None:
            raise RuntimeError(
                "nlsyms is required to remove tags: pass the recipe's symbol "
                "inventory, the same one tokenizer.nlsyms names. Set "
                "remove_tags: false to score the tags as text instead."
            )
        self.pattern = tag_pattern(load_symbols(nlsyms)) if remove_tags else None

    def _ensure_jiwer(self) -> None:
        """Raise with an install hint rather than an ImportError at scoring time."""
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute error rates. "
                "Please install it with `pip install espnet[asr]`."
            )

    def _pair(self, data: Dict[str, Path]) -> Tuple[List[str], List[str]]:
        """Return the cleaned ``(refs, hyps)`` for every row."""
        refs: List[str] = []
        hyps: List[str] = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref, hyp = row[self.ref_key], row[self.hyp_key]
            if self.pattern is not None:
                # Both sides: hyp_key may name the tagged text, and
                # stripping one side only turns a match into a total miss.
                ref = strip_tags(ref, self.pattern)
                hyp = strip_tags(hyp, self.pattern)
            refs.append(self._blank_safe(self.ref_cleaner(ref)))
            hyps.append(self._blank_safe(self.hyp_cleaner(hyp)))
        return refs, hyps

    @staticmethod
    def _blank_safe(text: str) -> str:
        """jiwer rejects an empty reference, so stand in for one."""
        stripped = text.strip()
        return stripped if stripped else "."

    def units(self, texts: List[str]) -> List[str]:
        """Return each text as the unit this metric counts."""
        raise NotImplementedError

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Score every row and write the alignment beside them."""
        self._ensure_jiwer()
        refs, hyps = self._pair(data)
        if not refs:
            logger.info("[%s] no rows; skipping %s", test_name, self.KEY)
            return {}

        refs, hyps = self.units(refs), self.units(hyps)
        score = jiwer.wer(refs, hyps) * 100
        details = jiwer.process_words(refs, hyps)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        alignment = test_dir / f"{self.KEY.lower()}_alignment"
        with alignment.open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {self.KEY: round(score, 2)}


class WER(ErrorRate):
    """Word error rate, s2t.sh's ``--token_type word`` pass."""

    KEY = "WER"

    def units(self, texts: List[str]) -> List[str]:
        """Words are whitespace-delimited, as espnet2's WordTokenizer has it."""
        return texts


class CER(ErrorRate):
    """Character error rate, s2t.sh's ``--token_type char`` pass."""

    KEY = "CER"

    #: What espnet2's CharTokenizer calls a space, and what s2t.sh's char pass
    #: therefore counts. Dropping spaces instead would score "thequick fox"
    #: against "the quick fox" as a perfect match.
    SPACE = "<space>"

    def units(self, texts: List[str]) -> List[str]:
        """Characters, with spaces as their own token, space-joined for jiwer."""
        return [" ".join(self.SPACE if c == " " else c for c in text) for text in texts]


class TER(ErrorRate):
    """Token error rate over BPE pieces, s2t.sh's ``--token_type bpe`` pass.

    Not sacreBLEU's TER (translation edit rate), which
    ``espnet3/systems/esp2_st`` reports under the same name. s2t.sh's ``ter``
    is an error rate over the model's own subword units.

    s2t.sh cannot remove non-linguistic symbols here even in principle --
    ``build_tokenizer.py:38-40`` raises for ``token_type=bpe`` -- so the
    reference's tags are always counted, and they tokenize as single pieces
    because they are the tokenizer's ``user_defined_symbols``.
    """

    KEY = "TER"

    def __init__(self, bpemodel: str | Path, **kwargs) -> None:
        """Initialize.

        Args:
            bpemodel: Path to the SentencePiece model the recipe trained.
            **kwargs: Forwarded to :class:`ErrorRate`.
        """
        super().__init__(**kwargs)
        self.bpemodel = str(bpemodel)
        self._tokenizer: Optional[object] = None

    def units(self, texts: List[str]) -> List[str]:
        """BPE pieces, space-joined so jiwer counts pieces rather than words."""
        if self._tokenizer is None:
            from espnet2.text.build_tokenizer import build_tokenizer

            self._tokenizer = build_tokenizer(token_type="bpe", bpemodel=self.bpemodel)
        return [" ".join(self._tokenizer.text2tokens(text)) for text in texts]

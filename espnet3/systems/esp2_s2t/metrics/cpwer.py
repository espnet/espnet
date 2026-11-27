"""Utterance-group cpWER.

This is the concatenated minimum-permutation WER (cpWER) of the CHiME-6
challenge, computed for each utterance group instead of each session. In each
group, the hypothesis speaker blocks are assigned to the reference speaker
blocks so that the total word error is smallest, and the error counts are
then summed over all groups.

Session-level cpWER uses one assignment for a whole meeting. Finding the
assignment within each short group is easier, so these numbers cannot be
compared with published session-level results. The class name and the
reported key include ``utterance group`` for this reason.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    from meeteval.wer import cp_word_error_rate
except ImportError:
    cp_word_error_rate = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")  # Whisper special tokens, e.g. <|1.20|>


def _strip_special_tokens(text: str) -> str:
    """Remove Whisper special tokens, such as timestamps and <|endoftext|>."""
    return re.sub(r"\s+", " ", _SPECIAL_RE.sub(" ", text)).strip()


def split_speakers(text: str, cleaner, speaker_change_symbol: str) -> List[str]:
    """Split SOT text into one text block per speaker.

    The text is split on ``speaker_change_symbol``, the symbol that the
    checkpoint was trained with. Whisper special tokens are removed, the
    cleaner is applied, and empty blocks are dropped.

    Args:
        text: Serialized SOT text, with or without timestamps.
        cleaner: A ``TextCleaner``, another function from str to str, or None.
        speaker_change_symbol: The separator between speaker blocks.

    Returns:
        One text block per speaker, in the order of the text.

    Examples:
        >>> text = "<|0.00|> hi<|0.40|> ???? <|0.20|> yes<|0.60|>"
        >>> split_speakers(text, None, "????")
        ['hi', 'yes']
    """
    blocks = []
    for chunk in text.split(speaker_change_symbol):
        chunk = _strip_special_tokens(chunk)
        if cleaner is not None:
            chunk = cleaner(chunk).strip()
        if chunk:
            blocks.append(chunk)
    return blocks


def _count_group_errors(ref_blocks: List[str], hyp_blocks: List[str]) -> Dict[str, int]:
    """Return the error counts of one utterance group under the best assignment.

    meeteval assigns the hypothesis speaker blocks to the reference speaker
    blocks so that the total word error is smallest, and returns the error
    counts for that assignment. It also handles blocks without a partner. An
    extra reference speaker counts as deletions, and an extra hypothesis
    speaker counts as insertions.

    Note:
        A minimal alignment can split the errors in more than one way. For
        example, the reference "a b" and the hypothesis "b c" have two errors,
        either as two substitutions or as one deletion and one insertion.
        meeteval and another Levenshtein implementation can then give
        different counts with the same total. The reported rate uses only the
        total, and the split appears only in the report files.

    Args:
        ref_blocks: One reference text block per speaker.
        hyp_blocks: One hypothesis text block per speaker.

    Returns:
        ``{"cor", "sub", "del", "ins", "ref_len"}`` word counts.
    """
    if not ref_blocks and not hyp_blocks:
        return {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}

    result = cp_word_error_rate(
        list(ref_blocks) or [""],
        list(hyp_blocks) or [""],
        # The blocks have no timestamps and are already in the serialized
        # speaker order. Without these flags, meeteval warns for every
        # utterance that it cannot sort them.
        reference_sort=False,
        hypothesis_sort=False,
    )
    return {
        # ref_len = cor + sub + del, so the number of correct words is
        # computed from the other counts.
        "cor": result.length - result.substitutions - result.deletions,
        "sub": result.substitutions,
        "del": result.deletions,
        "ins": result.insertions,
        "ref_len": result.length,
    }


def _get_percentage(numerator: float, denominator: float) -> float:
    """Return a percentage, or 0.0 when there is nothing to divide by."""
    return 100.0 * numerator / denominator if denominator else 0.0


def _get_utt_cpwer(errors: int, ref_len: int):
    """Return the cpWER of one group, or ``None`` when its reference is empty.

    ``_get_percentage`` returns 0.0 for a zero denominator, which would show a
    group with an empty reference and a non-empty hypothesis as perfect. The
    insertions of such a group still count in the total. Only the cpWER of the
    group is undefined, so it is reported as ``null``.
    """
    return _get_percentage(errors, ref_len) if ref_len else None


# Order of the fields in each row of the ``groups`` list.
_ROW = ("num_ref_speakers", "num_hyp_speakers", "cor", "sub", "del", "ins", "ref_len")


class UtteranceGroupCpWER(BaseMetric):
    """cpWER computed for each utterance group, reported as ``ug_cpWER``.

    Each utterance group has one text block per speaker, separated by
    ``speaker_change_symbol``. The order of the blocks is arbitrary, so each
    group is scored under the assignment of hypothesis blocks to reference
    blocks that gives the smallest word error. A session-level cpWER uses one
    assignment for a whole meeting, which is harder, so the two cannot be
    compared.

    Both inputs are SOT text. Timestamps and other Whisper special tokens are
    removed before scoring, so the hypothesis can keep them for DER.

    Args:
        speaker_change_symbol: The symbol that the model writes between
            speakers. It is required because it depends on the checkpoint.
            With the wrong symbol, every group would stay one block, and no
            error would be raised.
        clean_types: TextCleaner pipeline.

    Examples:
        >>> metric = UtteranceGroupCpWER("????")
        >>> metric.update(ref=["<|0.00|> a b ???? <|0.00|> c<|1.00|>"],
        ...               hyp=["<|0.00|> c ???? <|0.00|> a x<|1.00|>"])
        >>> metric.compute()
        {'ug_cpWER': 33.33}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ug_cpWER", "number"),)

    def __init__(self, speaker_change_symbol: str, clean_types=None) -> None:
        """Build the metric. See the class docstring for the arguments."""
        if not speaker_change_symbol:
            raise ValueError(
                "speaker_change_symbol is required and must be the symbol that "
                "the checkpoint was trained with. Without it, each utterance "
                "group is scored as one speaker, which gives bad scores and no "
                "error."
            )
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        self.add_state("errors", 0, "sum")
        self.add_state("ref_len", 0, "sum")
        # One _ROW for each group, kept only when the report files are written.
        self.add_state("groups", [], "cat")

    def _ensure_meeteval(self) -> None:
        """Raise an error if the optional meeteval dependency is missing.

        Raises:
            RuntimeError: If ``meeteval`` is not installed.
        """
        if cp_word_error_rate is None:
            raise RuntimeError(
                "meeteval is required to compute cpWER. "
                "Please install it with `pip install espnet[s2t]`."
            )

    def __call__(self, data, test_name, output_dir=None):
        """Score one test set from its SCP files, as the ``measure`` stage does.

        This works like ``BaseMetric.__call__``, but it keeps ``data``, so that
        :meth:`write_artifacts` can use the utterance ids, which :meth:`update`
        does not receive.
        """
        self._data = data
        return super().__call__(data, test_name, output_dir)

    def update(self, *, ref, hyp) -> None:
        """Add the error counts of each utterance group in the batch."""
        self._ensure_meeteval()
        for reference, hypothesis in zip(ref, hyp):
            ref_blocks = split_speakers(
                reference, self.cleaner, self.speaker_change_symbol
            )
            hyp_blocks = split_speakers(
                hypothesis, self.cleaner, self.speaker_change_symbol
            )
            counts = _count_group_errors(ref_blocks, hyp_blocks)
            self.errors += counts["sub"] + counts["del"] + counts["ins"]
            self.ref_len += counts["ref_len"]
            if self.collect_artifacts:
                counts["num_ref_speakers"] = len(ref_blocks)
                counts["num_hyp_speakers"] = len(hyp_blocks)
                self.groups.append([counts[key] for key in _ROW])

    def compute(self):
        """Return the summed errors over the summed reference words.

        Raises:
            ValueError: When every reference is empty. Otherwise the zero
                denominator would give a perfect 0.00%.
        """
        if self.ref_len == 0:
            raise ValueError(
                "cpWER cannot be computed because every reference is empty "
                "(the total ref_len is 0). Check that the ref input is bound "
                "to the "
                "reference text, for example dataset:text."
            )
        return {"ug_cpWER": round(_get_percentage(self.errors, self.ref_len), 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the per-group and by-speaker-count reports.

        Args:
            test_dir: Directory for the two JSON files.
        """
        utt_ids = [utt_id for utt_id, _ in self.iter_inputs(self._data, "hyp")]
        if len(utt_ids) != len(self.groups):
            raise RuntimeError(
                f"{len(self.groups)} scored groups for {len(utt_ids)} hypotheses"
            )
        per_utt = {}
        by_nspk = defaultdict(
            lambda: {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
        )
        for utt_id, row in zip(utt_ids, self.groups):
            group = dict(zip(_ROW, row))
            errors = group["sub"] + group["del"] + group["ins"]
            per_utt[utt_id] = {
                "cpwer": _get_utt_cpwer(errors, group["ref_len"]),
                "errors": errors,
                "ref_len": group["ref_len"],
                "num_ref_speakers": group["num_ref_speakers"],
                "num_hyp_speakers": group["num_hyp_speakers"],
            }
            for key in by_nspk[group["num_ref_speakers"]]:
                by_nspk[group["num_ref_speakers"]][key] += group[key]

        with (test_dir / "cpwer_per_utt.json").open("w", encoding="utf-8") as f:
            json.dump(per_utt, f, indent=2)
        with (test_dir / "cpwer_by_num_speakers.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    str(n): {
                        **counts,
                        "cpwer": _get_utt_cpwer(
                            counts["sub"] + counts["del"] + counts["ins"],
                            counts["ref_len"],
                        ),
                    }
                    for n, counts in sorted(by_nspk.items())
                },
                f,
                indent=2,
            )

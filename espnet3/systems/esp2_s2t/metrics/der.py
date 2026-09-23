"""Utterance-group Diarization Error Rate.

Hypothesis timing comes from the model's own inline Whisper ``<|t|>``
timestamps; reference timing comes from the inline timestamps in the
reference SOT text. Each speaker block (delimited by the speaker-change
symbol) is treated as one speaker; within a block, consecutive
``<|start|> ... <|end|>`` pairs become segments.

Timestamps are relative to each utterance-group window, so **every group is
scored as its own RTTM "file"** and the total is aggregated by SCTK's
``md-eval.pl``. A meeting split across many groups is never scored as one
long recording, so these numbers are not comparable with published
session-level ones. The class and the reported key both say ``utterance
group`` for that reason.
"""

import json
import logging
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric

from espnet3.systems.esp2_s2t.metrics.cpwer import split_speakers

logger = logging.getLogger(__name__)

# The speaker-change symbol belongs to the checkpoint, so it is a constructor
# argument rather than a constant: a metric cannot know which symbol a model
# was trained with.
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_TS_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_DER_RE = re.compile(r"OVERALL SPEAKER DIARIZATION ERROR\s*=\s*([\d.]+)\s*percent")


def segments_from_sot(
    text: str, speaker_change_symbol: str
) -> List[Tuple[int, float, float]]:
    """Parse SOT text into ``(speaker_index, start, end)`` segments.

    Each speaker-change-delimited block is one synthetic speaker, numbered by
    its position in the text (0-based). Within a block, timestamps are
    paired ``(start, end)`` in order of appearance. A zero/negative-length
    pair is dropped, and so is an unpaired trailing timestamp, which a
    decode produces when the model opens a segment and reaches its length
    limit before closing it. That tolerance is aimed at hypotheses, not
    references: across the AMI test set the reference carries no unpaired
    timestamp in any of its 9319 blocks, while the recorded decode carries
    one.
    """
    text = text.replace(speaker_change_symbol, _SEP)
    segments = []
    for idx, block in enumerate(text.split(_SEP)):
        timestamps = [float(x) for x in _TS_RE.findall(block)]
        for k in range(0, len(timestamps) - 1, 2):
            start, end = timestamps[k], timestamps[k + 1]
            if end > start:
                segments.append((idx, start, end))
    return segments


def write_rttm(rows: List[Tuple[str, int, float, float]], path) -> None:
    """Write ``(utt_id, speaker_index, start, end)`` rows as an RTTM file.

    Field order matches SCTK's md-eval.pl expectation:
    ``SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA>
    <NA>``. Each ``utt_id`` is scored by md-eval.pl as its own "file"
    (recording), so every AMI utterance group becomes an independent
    diarization instance.
    """
    with Path(path).open("w", encoding="utf-8") as f:
        for utt_id, speaker_index, start, end in rows:
            f.write(
                f"SPEAKER {utt_id} 1 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> spk{speaker_index} <NA> <NA>\n"
            )


def find_md_eval() -> str:
    """Return the path to SCTK's md-eval.pl inside this checkout.

    Raises:
        FileNotFoundError: When SCTK has not been built inside this
            checkout. There is no fallback: scoring against an unknown
            md-eval would make the number unreproducible.
    """
    return _find_md_eval_from(Path(__file__).resolve())


def _find_md_eval_from(start: Path) -> str:
    """Search ``start``'s ancestors for ``tools/sctk/bin/md-eval.pl``.

    ``Path.parents`` does not stop at a repository boundary, so an unbounded
    walk could climb past this checkout and return an md-eval.pl from some
    other checkout entirely -- there are several on this machine. The walk
    instead stops at, and includes, the first ancestor holding ``.git``
    (this checkout's root): that directory is still checked for
    ``tools/sctk``, but its own parent is never examined.

    Args:
        start: File to search from, typically this module's own path.

    Returns:
        The resolved path to ``md-eval.pl``.

    Raises:
        FileNotFoundError: When no ``tools/sctk/bin/md-eval.pl`` is found at
            or below the repository root.
    """
    root = None
    for parent in start.parents:
        candidate = parent / "tools" / "sctk" / "bin" / "md-eval.pl"
        if candidate.is_file():
            return str(candidate)
        if (parent / ".git").exists():
            root = parent
            break
    where = f" in the {root} checkout" if root is not None else ""
    raise FileNotFoundError(
        f"md-eval.pl not found{where}. Build it with "
        "tools/installers/install_sctk.sh."
    )


def run_md_eval(md_eval: str, ref_rttm: str, hyp_rttm: str, collar: float) -> float:
    """Score two RTTM files with SCTK's md-eval.pl.

    Args:
        md_eval: Path to ``md-eval.pl``.
        ref_rttm: Reference RTTM path.
        hyp_rttm: Hypothesis RTTM path.
        collar: Forgiveness collar in seconds.

    Returns:
        The overall diarization error rate as a percentage.

    Raises:
        SystemExit: When md-eval.pl exits non-zero, or prints no overall
            line. Both are loud on purpose: a silently wrong DER is the
            failure mode this recipe has already been bitten by.
    """
    proc = subprocess.run(
        [md_eval, "-c", str(collar), "-r", ref_rttm, "-s", hyp_rttm],
        capture_output=True,
        text=True,
    )
    match = _DER_RE.search(proc.stdout)
    if proc.returncode != 0 or not match:
        logger.error(proc.stdout[-2000:])
        logger.error(proc.stderr[-1000:])
        if proc.returncode != 0:
            raise SystemExit(f"md-eval.pl exited with status {proc.returncode}.")
        raise SystemExit("Could not parse DER from md-eval.pl output.")
    return float(match.group(1))


class UtteranceGroupDER(BaseMetric):
    """DER over one utterance group at a time, reported as ``ug_DER``.

    Each utterance group holds one text block per speaker, separated by
    ``speaker_change_symbol``, with inline Whisper timestamps. Every group is
    written as its own RTTM "file" and scored together by SCTK's
    ``md-eval.pl``, matching cpWER's utterance-group scope: a meeting split
    across many groups is never scored as one long recording.

    Args:
        ref_key: Alias of the reference SCP input.
        hyp_key: Alias of the hypothesis SCP input.
        collar: Forgiveness collar in seconds, passed to md-eval.pl.
        md_eval: Path to ``md-eval.pl``. Resolved from this repository via
            ``find_md_eval`` when not given.
        clean_types: TextCleaner pipeline used only to count reference
            speakers for the by-speaker-count breakdown, so that breakdown
            covers the same groups as cpWER's.
        speaker_change_symbol: The symbol the model separates speakers with.
            It belongs to the checkpoint, so it has no useful default here.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        collar: float = 0.25,
        md_eval: str = None,
        clean_types=("whisper_en",),
        speaker_change_symbol: str = "????",
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.collar = collar
        self.md_eval = md_eval or find_md_eval()
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol

    def __call__(self, data, test_name, output_dir):
        """Score one test set.

        Args:
            data: Alias to path mapping. Needs ``self.ref_key`` and
                ``self.hyp_key``, both SCP files with matching utterance ids
                in the same order, carrying SOT text with inline timestamps
                (``hyp_sot``/``ref_sot`` in this recipe's config).
            test_name: Test set name, used for the side-file directory.
            output_dir: Root the side files are written under.

        Returns:
            ``{"ug_DER": <percentage rounded to two decimals>}``.
        """
        ref_segments: Dict[str, List[Tuple[int, float, float]]] = {}
        hyp_segments: Dict[str, List[Tuple[int, float, float]]] = {}
        num_ref_speakers: Dict[str, int] = {}
        num_dropped_empty_ref = 0
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref_text = row[self.ref_key]
            ref_seg = segments_from_sot(ref_text, self.speaker_change_symbol)
            if not ref_seg:
                # md-eval needs reference speech for a file to be scored.
                num_dropped_empty_ref += 1
                continue
            ref_segments[utt_id] = ref_seg
            hyp_segments[utt_id] = segments_from_sot(
                row[self.hyp_key], self.speaker_change_symbol
            )
            # Same speaker-count definition as cpWER (split_speakers), so the
            # two by-speaker-count tables describe the same set of groups.
            num_ref_speakers[utt_id] = len(
                split_speakers(ref_text, self.cleaner, self.speaker_change_symbol)
            )

        if num_dropped_empty_ref:
            logger.warning(
                "%d utterance group(s) had no reference segments (ref_sot "
                "carried no usable <start,end> timestamp pair) and were "
                "dropped from DER for test set %r.",
                num_dropped_empty_ref,
                test_name,
            )

        test_dir = Path(output_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        def rows_for(utt_ids, segments) -> List[Tuple[str, int, float, float]]:
            return [
                (utt_id, idx, start, end)
                for utt_id in utt_ids
                for idx, start, end in segments[utt_id]
            ]

        all_ids = sorted(ref_segments)
        ref_rttm = test_dir / "ref.rttm"
        hyp_rttm = test_dir / "hyp.rttm"
        write_rttm(rows_for(all_ids, ref_segments), ref_rttm)
        write_rttm(rows_for(all_ids, hyp_segments), hyp_rttm)
        score = run_md_eval(self.md_eval, str(ref_rttm), str(hyp_rttm), self.collar)

        by_nspk = defaultdict(list)
        for utt_id, n in num_ref_speakers.items():
            by_nspk[n].append(utt_id)

        der_by_nspk = {}
        for n in sorted(by_nspk):
            utt_ids = sorted(by_nspk[n])
            r = test_dir / f"ref_{n}spk.rttm"
            h = test_dir / f"hyp_{n}spk.rttm"
            write_rttm(rows_for(utt_ids, ref_segments), r)
            write_rttm(rows_for(utt_ids, hyp_segments), h)
            d = run_md_eval(self.md_eval, str(r), str(h), self.collar)
            der_by_nspk[n] = {"der": d, "num_groups": len(utt_ids)}

        with (test_dir / "der_by_num_speakers.json").open("w", encoding="utf-8") as f:
            json.dump({str(n): v for n, v in der_by_nspk.items()}, f, indent=2)

        return {"ug_DER": round(score, 2)}

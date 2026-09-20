"""DER metric for the AMI SOT recipe.

The hypothesis timing comes from the model's own inline Whisper ``<|t|>``
timestamps in ``hyp_sot``; the reference timing comes from the inline
timestamps in the reference SOT text (``ref_sot``). Each speaker block
(delimited by the speaker-change token) is treated as one speaker; within a
block, consecutive ``<|start|> ... <|end|>`` timestamp pairs become
segments. Timestamps are relative to each utterance-group window, so every
utterance group is scored as its own RTTM "file" and DER is aggregated by
SCTK's ``md-eval.pl`` with a 0.25 s collar, the same tool and collar used
by ESPnet diarization recipes. md-eval.pl is the reference implementation
of the score, so it is what the number means here; no external diarization
library is added.

``segments_from_sot``, ``write_rttm`` and ``run_md_eval`` are a port of the
scorer used by the ESPnet2 AMI SOT recipe. What the port has to stay
faithful to is the scoring definition above, not any one copy of that
script: the timestamp pairing, the RTTM construction, and the md-eval.pl
invocation with its collar. A regression test pins those by rescoring a
recorded decode of the full test set.

That recorded decode scores DER 8.57% here. The ESPnet2 recipe reports
8.33% for the same model decoded another way (the openai-whisper
``transcribe()`` path); this recipe decodes with ESPnet's own beam search
instead, see ``src/inference.py``. The two numbers belong to different
hypothesis sets, not to two different scorers.
"""

import importlib.util
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric

if __package__:
    # Normal case: this module is part of the real egs3.ami.s2t.src.metrics
    # package, so the relative import resolves against it directly. Gating on
    # __package__ instead of wrapping this in try/except ImportError means a
    # genuine breakage in cpwer.py (for example split_speakers renamed)
    # surfaces as its own ImportError here, rather than being caught and
    # silently rerouted into the fallback below.
    from .cpwer import split_speakers
else:
    # A relative import has no parent package to resolve against when this
    # file is loaded standalone, for example via
    # ``importlib.util.spec_from_file_location`` in tests with a flat,
    # non-dotted module name (so __package__ is ``""``). Fall back to loading
    # the sibling module by file path so ``split_speakers`` still comes from
    # cpwer.py rather than being redefined here. The synthetic name is kept
    # dot-free on purpose: a dotted name would give the loaded cpwer.py its
    # own non-empty (and misleading) __package__, which would then send
    # cpwer.py's own __package__ check down the wrong branch when it tries to
    # import separator.py.
    _module_key = f"{__name__}_ami_sot_cpwer_impl"
    if _module_key in sys.modules:
        _cpwer = sys.modules[_module_key]
    else:
        _spec = importlib.util.spec_from_file_location(
            _module_key, Path(__file__).resolve().parent / "cpwer.py"
        )
        _cpwer = importlib.util.module_from_spec(_spec)
        sys.modules[_module_key] = _cpwer
        _spec.loader.exec_module(_cpwer)
    split_speakers = _cpwer.split_speakers

if __package__:
    # Same reasoning as the split_speakers import above.
    from ..separator import SPEAKER_CHANGE_SYMBOL
else:
    # Unlike the cpwer.py shim above, this one is not cached in sys.modules:
    # SPEAKER_CHANGE_SYMBOL must reflect whatever separator.py's environment
    # variable is set to at the moment this module is (re)loaded, which is
    # exactly what a test reloading this module after monkeypatching the
    # environment relies on. separator.py has no heavy imports of its own, so
    # loading it fresh on every reload costs nothing worth caching for.
    _spec = importlib.util.spec_from_file_location(
        f"{__name__}_ami_sot_separator_impl",
        Path(__file__).resolve().parent.parent / "separator.py",
    )
    _separator = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_separator)
    SPEAKER_CHANGE_SYMBOL = _separator.SPEAKER_CHANGE_SYMBOL

logger = logging.getLogger(__name__)

# The model separates speakers with a single BPE token, resolved once for the
# whole recipe as SPEAKER_CHANGE_SYMBOL (separator.py). This recipe's
# inference step (src/inference.py) normalizes it to "<sc>" before writing
# hyp_sot and ref_sot text, so "<sc>" is what this metric normally sees; the
# raw separator is still accepted for text that has not gone through that
# normalization. Both inference.py and this constant import the same symbol
# from separator.py, so it cannot drift out of sync between them.
_SEP_VARIANTS = ("<sc>", SPEAKER_CHANGE_SYMBOL)
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_TS_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_DER_RE = re.compile(r"OVERALL SPEAKER DIARIZATION ERROR\s*=\s*([\d.]+)\s*percent")


def segments_from_sot(text: str) -> List[Tuple[int, float, float]]:
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
    for variant in _SEP_VARIANTS:
        text = text.replace(variant, _SEP)
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


class DER(BaseMetric):
    """Utterance-group Diarization Error Rate for SOT output.

    Each utterance group holds one text block per speaker, separated by
    ``<sc>``, with inline Whisper timestamps. Every group is written as its
    own RTTM "file" and scored together by SCTK's ``md-eval.pl``. This is
    utterance-group DER, matching cpWER's utterance-group scope: a single
    meeting split across many groups is not scored as one long recording.

    Args:
        ref_key: Alias of the reference SCP input.
        hyp_key: Alias of the hypothesis SCP input.
        collar: Forgiveness collar in seconds, passed to md-eval.pl.
        md_eval: Path to ``md-eval.pl``. Resolved from this repository via
            ``find_md_eval`` when not given.
        clean_types: TextCleaner pipeline used only to count reference
            speakers for the by-speaker-count breakdown, so that breakdown
            covers the same groups as cpWER's.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        collar: float = 0.25,
        md_eval: str = None,
        clean_types=("whisper_en",),
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.collar = collar
        self.md_eval = md_eval or find_md_eval()
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)

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
            ``{"DER": <percentage rounded to two decimals>}``.
        """
        ref_segments: Dict[str, List[Tuple[int, float, float]]] = {}
        hyp_segments: Dict[str, List[Tuple[int, float, float]]] = {}
        num_ref_speakers: Dict[str, int] = {}
        num_dropped_empty_ref = 0
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref_text = row[self.ref_key]
            ref_seg = segments_from_sot(ref_text)
            if not ref_seg:
                # md-eval needs reference speech for a file to be scored.
                num_dropped_empty_ref += 1
                continue
            ref_segments[utt_id] = ref_seg
            hyp_segments[utt_id] = segments_from_sot(row[self.hyp_key])
            # Same speaker-count definition as cpWER (split_speakers), so the
            # two by-speaker-count tables describe the same set of groups.
            num_ref_speakers[utt_id] = len(split_speakers(ref_text, self.cleaner))

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

        return {"DER": round(score, 2)}

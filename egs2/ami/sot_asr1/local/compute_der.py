#!/usr/bin/env python3
"""Compute utterance-group Diarization Error Rate (DER) for SOT outputs.

Mirrors the paper-side DER computation in
``TS-ASR-Whisper/src/utils/evaluation.py`` (uses
``pyannote.metrics.DiarizationErrorRate`` with ``collar=0.25``).

Reads SOT-format text files where each line is::

    <utt_id> <|t1|> text <|t2|> <sc> <|t1|> text <|t2|> <sc> ...

For each utterance, builds a pyannote ``Annotation`` for both reference and
hypothesis by splitting on the SOT separator and parsing the ``<|t|>``
timestamp markers per speaker block. The aggregate DER is reported, plus
a per-#speakers breakdown.

Usage::

    python local/compute_der.py \\
        --hyp_text_sot exp/converted_sst_small/decode_test_full/1best_recog/text_sot \\
        --ref_text_sot data/test/text \\
        --output_dir exp/converted_sst_small/decode_test_full/eval
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("compute_der")


_TS_RE = re.compile(r"<\|([\d.]+)\|>")
# Matches any Whisper-style ``<|...|>`` tag whose content is NOT a number
# (e.g. ``<|endoftext|>``, ``<|en|>``, ``<|transcribe|>``). Used to strip
# residual special tokens from reference/hypothesis strings BEFORE parsing
# so they aren't mis-attributed to a trailing speech segment.
_NON_TS_SPECIAL_RE = re.compile(r"<\|(?!\d+\.\d+\|)[^|]*\|>")


def _clean_specials(s: str) -> str:
    """Strip non-timestamp Whisper special tokens (``<|endoftext|>`` etc.).
    Keeps timestamp tokens (``<|0.00|>``...) intact."""
    s = _NON_TS_SPECIAL_RE.sub("", s)
    return " ".join(s.split())


def parse_string_to_objects(s: str, duration: Optional[float] = None) -> List[dict]:
    """Extract (start, end, text) tuples from a Whisper SOT timestamped
    string. Adapted from TS-ASR-Whisper/src/utils/evaluation.py."""
    times = _TS_RE.findall(s)
    text_segments = _TS_RE.split(s)[1:]
    objects: List[dict] = []
    for i in range(len(times) - 1):
        start = float(times[i])
        end = float(times[i + 1])
        text = text_segments[2 * i + 1].strip()
        if text:
            objects.append({"start": start, "end": end, "text": text})
    if times and duration is not None:
        last_idx = len(times) - 1
        trailing_idx = 2 * last_idx + 1
        if trailing_idx < len(text_segments):
            trailing = text_segments[trailing_idx].strip()
            if trailing:
                objects.append(
                    {"start": float(times[last_idx]), "end": duration, "text": trailing}
                )
    return objects


def sot_to_annotation(
    sot_string: str,
    session_id: str,
    separator: str = "<sc>",
    duration: Optional[float] = None,
):
    """Convert an SOT string into a pyannote ``Annotation`` with positional
    speaker labels (``spk_0``, ``spk_1``, ...). pyannote's DER uses optimal
    speaker mapping (Hungarian), so absolute label identity doesn't matter.

    Strips non-timestamp Whisper special tokens (``<|endoftext|>`` etc.)
    before parsing so a residual ``<|endoftext|>`` after the last
    timestamp doesn't get mis-attributed as trailing speech.
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=session_id)
    blocks = _clean_specials(sot_string).split(separator)
    for spk_idx, block in enumerate(blocks):
        speaker = f"spk_{spk_idx}"
        for seg in parse_string_to_objects(block, duration=duration):
            if seg["end"] > seg["start"]:
                annotation[Segment(seg["start"], seg["end"]), speaker] = speaker
    return annotation


def infer_duration(sot_string: str, fallback: float = 30.0) -> float:
    times = _TS_RE.findall(sot_string)
    if not times:
        return fallback
    return max(float(t) for t in times)


def load_text_file(path: Path) -> Dict[str, str]:
    """Load a Kaldi-format text file (``utt_id rest_of_line``)."""
    out: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
            else:
                out[parts[0]] = ""
    return out


def num_ref_speakers(ref_sot: str, separator: str = "<sc>") -> int:
    return sum(1 for blk in ref_sot.split(separator) if blk.strip())


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hyp_text_sot", required=True, type=Path)
    p.add_argument(
        "--ref_text_sot",
        required=True,
        type=Path,
        help="Reference SOT text (e.g. data/test/text).",
    )
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--separator", default="<sc>")
    p.add_argument("--collar", type=float, default=0.25)
    args = p.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    hyp = load_text_file(args.hyp_text_sot)
    ref = load_text_file(args.ref_text_sot)
    common = sorted(set(hyp) & set(ref))
    if not common:
        raise SystemExit("No common utt_ids between hyp and ref!")
    miss_h = set(ref) - set(hyp)
    miss_r = set(hyp) - set(ref)
    if miss_h:
        logger.warning(f"{len(miss_h)} ref utts missing from hyp")
    if miss_r:
        logger.warning(f"{len(miss_r)} hyp utts missing from ref")
    logger.info(f"Evaluating DER on {len(common)} common utterances")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric = DiarizationErrorRate(collar=args.collar)
    by_nspk_components: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"fa": 0.0, "md": 0.0, "conf": 0.0, "total": 0.0}
    )
    nspk_groups: Dict[int, List[str]] = defaultdict(list)
    skipped = 0
    per_utt: Dict[str, dict] = {}
    total_fa = total_md = total_conf = total_dur = 0.0
    for uid in common:
        r_str = ref.get(uid, "")
        h_str = hyp.get(uid, "")
        n_spk = num_ref_speakers(r_str, separator=args.separator)
        nspk_groups[n_spk].append(uid)
        try:
            D = max(infer_duration(r_str), infer_duration(h_str), 1.0)
            r_ann = sot_to_annotation(r_str, uid, separator=args.separator, duration=D)
            h_ann = sot_to_annotation(h_str, uid, separator=args.separator, duration=D)
            if len(r_ann) == 0:
                skipped += 1
                continue
            components = metric.compute_components(r_ann, h_ann)
            fa = float(components.get("false alarm", 0.0))
            md = float(components.get("missed detection", 0.0))
            conf = float(components.get("confusion", 0.0))
            total = float(components.get("total", 0.0))
            total_fa += fa
            total_md += md
            total_conf += conf
            total_dur += total
            by_nspk_components[n_spk]["fa"] += fa
            by_nspk_components[n_spk]["md"] += md
            by_nspk_components[n_spk]["conf"] += conf
            by_nspk_components[n_spk]["total"] += total
            per_utt[uid] = {
                "fa": fa,
                "md": md,
                "conf": conf,
                "total": total,
                "n_ref_spk": n_spk,
            }
        except Exception as e:
            skipped += 1
            logger.warning(f"DER failed for {uid}: {e}")

    if skipped:
        logger.info(f"Skipped {skipped} utterances (empty ref or pyannote error)")

    # Aggregate DER = (FA + MD + CONF) / TOTAL across all utts.
    der_value = (total_fa + total_md + total_conf) / max(total_dur, 1e-9)
    der_by_n = {
        n: {
            "der": (c["fa"] + c["md"] + c["conf"]) / max(c["total"], 1e-9),
            "num_sessions": len(nspk_groups[n]),
            "fa": c["fa"],
            "md": c["md"],
            "conf": c["conf"],
            "total": c["total"],
        }
        for n, c in sorted(by_nspk_components.items())
    }

    logger.info(
        f"DER (collar={args.collar}, {len(common)-skipped} utts): "
        f"{der_value*100:.2f}%"
    )
    for n in sorted(der_by_n):
        logger.info(
            f"  {n} spk(s): DER={der_by_n[n]['der']*100:.2f}%  "
            f"(n={der_by_n[n]['num_sessions']})"
        )

    with open(args.output_dir / "der.json", "w") as f:
        json.dump(
            {
                "der": der_value,
                "collar": args.collar,
                "n_utts": len(common) - skipped,
                "n_skipped": skipped,
            },
            f,
            indent=2,
        )
    with open(args.output_dir / "der_by_num_speakers.json", "w") as f:
        json.dump(der_by_n, f, indent=2)
    with open(args.output_dir / "per_utt_der.json", "w") as f:
        json.dump(per_utt, f, indent=2)


if __name__ == "__main__":
    main()

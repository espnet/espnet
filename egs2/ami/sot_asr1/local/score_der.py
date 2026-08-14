#!/usr/bin/env python3
"""Utterance-group Diarization Error Rate (DER) for SOT multi-talker output.

The hypothesis timing comes from the model's own inline Whisper ``<|t|>``
timestamps in ``text_sot``; the reference timing comes from the inline
timestamps in the reference SOT text (``data/<set>/text``). Each speaker block
(delimited by the speaker-change token) is treated as one speaker; within a
block, consecutive ``<|start|> ... <|end|>`` timestamp pairs become segments.
Timestamps are relative to each utterance-group window, so every utterance
group is scored as its own RTTM "file" and DER is aggregated by SCTK's
``md-eval.pl`` (collar 0.25 s), the same tool used by ESPnet diarization
recipes. No external diarization library is added.

Usage:
    python local/score_der.py \
        --hyp_text_sot <decode_dir>/1best_recog/text_sot \
        --ref_text data/test/text \
        --output_dir <out_dir> \
        --collar 0.25
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("score_der")

_SEP_VARIANTS = ("<sc>", "????")
_SEP = "▁SPKCHANGE▁"
_TS_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_DER_RE = re.compile(r"OVERALL SPEAKER DIARIZATION ERROR\s*=\s*([\d.]+)\s*percent")


def load_text(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


def parse_segments(text: str) -> List[Tuple[str, float, float]]:
    """Parse SOT text into (speaker_label, start, end) segments.

    Each speaker-change-delimited block is one synthetic speaker; within a
    block, timestamps are paired (start, end). Zero/negative-length or unpaired
    trailing timestamps are dropped.
    """
    for v in _SEP_VARIANTS:
        text = text.replace(v, _SEP)
    segments = []
    for idx, block in enumerate(text.split(_SEP)):
        ts = [float(x) for x in _TS_RE.findall(block)]
        for k in range(0, len(ts) - 1, 2):
            start, end = ts[k], ts[k + 1]
            if end > start:
                segments.append((f"spk{idx}", start, end))
    return segments


def write_rttm(path: str, seg_by_utt: Dict[str, List[Tuple[str, float, float]]]):
    with open(path, "w") as f:
        for utt in sorted(seg_by_utt):
            for spk, start, end in seg_by_utt[utt]:
                f.write(
                    f"SPEAKER {utt} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n"
                )


def find_md_eval() -> str:
    """Locate SCTK's md-eval.pl whether or not path.sh has been sourced."""
    cand = shutil.which("md-eval.pl")
    if cand:
        return cand
    roots = []
    if os.environ.get("MAIN_ROOT"):
        roots.append(os.environ["MAIN_ROOT"])
    # Repo root relative to this file: egs2/<corpus>/sot_asr1/local/ -> up 4.
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
    for r in roots:
        p = os.path.join(r, "tools", "sctk", "bin", "md-eval.pl")
        if os.path.isfile(p):
            return p
    raise SystemExit(
        "md-eval.pl not found. Install SCTK via tools/installers/install_sctk.sh "
        "(or source tools/extra_path.sh)."
    )


def run_md_eval(md_eval: str, ref_rttm: str, hyp_rttm: str, collar: float) -> float:
    proc = subprocess.run(
        [md_eval, "-c", str(collar), "-r", ref_rttm, "-s", hyp_rttm],
        capture_output=True,
        text=True,
    )
    m = _DER_RE.search(proc.stdout)
    if not m:
        logger.error(proc.stdout[-2000:])
        logger.error(proc.stderr[-1000:])
        raise SystemExit("Could not parse DER from md-eval.pl output.")
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Utterance-group DER for SOT output.")
    parser.add_argument("--hyp_text_sot", required=True, help="Hyp text_sot file.")
    parser.add_argument("--ref_text", required=True, help="Reference text file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--md_eval", default=None, help="Path to md-eval.pl.")
    args = parser.parse_args()

    md_eval = args.md_eval or find_md_eval()
    os.makedirs(args.output_dir, exist_ok=True)

    hyp_raw = load_text(args.hyp_text_sot)
    ref_raw = load_text(args.ref_text)
    common = sorted(set(hyp_raw) & set(ref_raw))
    if not common:
        raise SystemExit("No common utterance ids between hypothesis and reference.")

    ref_seg, hyp_seg, n_ref_spk = {}, {}, {}
    for utt in common:
        rseg = parse_segments(ref_raw[utt])
        hseg = parse_segments(hyp_raw[utt])
        n = len({s for s, _, _ in rseg})
        if not rseg:  # md-eval needs reference speech for a file to be scored
            continue
        ref_seg[utt] = rseg
        hyp_seg[utt] = hseg
        n_ref_spk[utt] = n

    # --- overall DER ---
    ref_rttm = os.path.join(args.output_dir, "ref.rttm")
    hyp_rttm = os.path.join(args.output_dir, "hyp.rttm")
    write_rttm(ref_rttm, ref_seg)
    write_rttm(hyp_rttm, hyp_seg)
    der = run_md_eval(md_eval, ref_rttm, hyp_rttm, args.collar)
    logger.info(f"DER (collar={args.collar}): {der:.2f}%  ({len(ref_seg)} groups)")

    # --- DER by reference speaker count ---
    buckets = defaultdict(list)
    for utt, n in n_ref_spk.items():
        buckets[n].append(utt)
    der_by_nspk = {}
    for n in sorted(buckets):
        utts = buckets[n]
        r = os.path.join(args.output_dir, f"ref_{n}spk.rttm")
        h = os.path.join(args.output_dir, f"hyp_{n}spk.rttm")
        write_rttm(r, {u: ref_seg[u] for u in utts})
        write_rttm(h, {u: hyp_seg[u] for u in utts})
        d = run_md_eval(md_eval, r, h, args.collar)
        der_by_nspk[n] = {"der": d, "num_groups": len(utts)}
        logger.info(f"  {n}-spk DER: {d:.2f}%  (n={len(utts)})")

    with open(os.path.join(args.output_dir, "der.json"), "w") as f:
        json.dump(
            {"der": der, "collar": args.collar, "num_groups": len(ref_seg)}, f, indent=2
        )
    with open(os.path.join(args.output_dir, "der_by_num_speakers.json"), "w") as f:
        json.dump(der_by_nspk, f, indent=2)
    logger.info(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()

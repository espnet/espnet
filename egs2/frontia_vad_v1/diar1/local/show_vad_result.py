#!/usr/bin/env python3
"""VAD results of a diarization experiment, at one setting chosen on a dev set.

    python3 local/show_vad_result.py DIAR_EXP [--tune_set ami_mh_dev] [--collar 0.0]

Stage 7 (scripts/utils/score_der.sh) scores every set at thresholds 0.3-0.7
and median filters 1 and 11, and RESULTS.md reports each set's best setting,
picked on that set itself. This picks the setting (threshold, median) with the
lowest error on --tune_set and reports every set at it, next to each set's own
best. With one pseudo speaker in reference and hypothesis there is no speaker
confusion, so the diarization error is the detection error: (missed + false
alarm speech) / reference speech.

Prints a Markdown table (also written to DIAR_EXP/RESULTS_vad.md).
"""

import argparse
import re
from pathlib import Path

RESULT_RE = re.compile(r"result_th([0-9.]+)_med(\d+)_collar")


def read_md_eval(path):
    """Scored, missed and false-alarm speaker time (s) of an md-eval output."""
    text = path.read_text()

    def seconds(name):
        m = re.search(rf"{name}\s*=\s*([0-9.]+)\s*secs", text)
        return float(m.group(1)) if m else None

    scored, miss, fa = (
        seconds("SCORED SPEAKER TIME"),
        seconds("MISSED SPEAKER TIME"),
        seconds("FALARM SPEAKER TIME"),
    )
    if scored is None or miss is None or fa is None:
        return None
    return scored, miss, fa


def read_set(scoring_dir, collar):
    """(threshold, median) -> (scored s, missed s, false alarm s)."""
    out = {}
    for path in scoring_dir.glob(f"result_th*_med*_collar{collar}"):
        m = RESULT_RE.match(path.name)
        r = read_md_eval(path) if m else None
        if r:
            out[(float(m.group(1)), int(m.group(2)))] = r
    return out


def pct(r):
    scored, miss, fa = r
    return 100 * (miss + fa) / scored, 100 * fa / scored, 100 * miss / scored


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("diar_exp", type=Path)
    ap.add_argument("--tune_set", default="ami_mh_dev")
    ap.add_argument("--collar", default="0.0")
    a = ap.parse_args()

    sets = {}
    for d in sorted(a.diar_exp.glob("diarized_*/scoring")):
        r = read_set(d, a.collar)
        if r:
            sets[d.parent.name[len("diarized_") :]] = r
    if a.tune_set not in sets:
        raise SystemExit(f"no scores of {a.tune_set} in {a.diar_exp}")
    tune = sets[a.tune_set]
    setting = min(tune, key=lambda k: pct(tune[k])[0])
    th, med = setting

    lines = [
        f"# VAD results: {a.diar_exp.name}",
        "",
        "Detection error (%) = (missed + false alarm speech) / reference speech,"
        f" md-eval, collar {a.collar}.",
        f"Tuned on {a.tune_set}: threshold {th}, median filter {med}.",
        "",
        "| set | speech h | DetER (tuned) | FA | Miss"
        " | DetER (set's own best) | its threshold, median |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in sets.items():
        own = min(r, key=lambda k: pct(r[k])[0])
        if setting in r:
            d, fa, miss = pct(r[setting])
            tuned = f"{d:.2f} | {fa:.2f} | {miss:.2f}"
        else:
            tuned = "— | — | —"
        hours = r[own][0] / 3600
        best = f"{pct(r[own])[0]:.2f} | {own[0]}, {own[1]}"
        lines.append(f"| {name} | {hours:.1f} | {tuned} | {best} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (a.diar_exp / "RESULTS_vad.md").write_text(text)


if __name__ == "__main__":
    main()

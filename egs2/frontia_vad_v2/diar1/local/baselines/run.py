#!/usr/bin/env python3
"""Speech probabilities of an external VAD model on test sets (one task of an array).

    python3 local/baselines/run.py --model silero --out exp/baselines \
        --sets dump/raw/<set> ... [--task 1 --ntasks 1] [--model_dir <offline weights>]

The recordings of all sets, longest first, go round robin to the --ntasks tasks. Writes
<out>/<model>/diar_eval/diarized_<set>/predictions/<recording>.npy (one probability per
frame) and diarize.<task>.scp, the layout of local/v2/vad_infer.py, and
<out>/<model>/frames.json (frame shift and offset, the tuning grid). Audio: wav.scp
entries are Omni-IO entries (<bin>:<start byte>:<end byte>, decoded here without ESPnet)
or audio files; the first channel, resampled to 16 kHz.
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import FS, MODELS  # noqa: E402

ENTRY = re.compile(r"^(.+):(\d+):(\d+)$")


def load(entry: str):
    m = ENTRY.match(entry)
    if m and not Path(entry).exists():
        with open(m.group(1), "rb") as f:
            f.seek(int(m.group(2)))
            data = f.read(int(m.group(3)) - int(m.group(2)))
        x, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    else:
        x, rate = soundfile.read(entry, dtype="float32", always_2d=True)
    x = x[:, 0]
    if rate != FS:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(rate, FS)
        x = resample_poly(x, FS // g, rate // g).astype(np.float32)
    return x


def table(path):
    return dict(x.split(maxsplit=1) for x in Path(path).read_text().splitlines() if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--model_dir", default=None)
    a = ap.parse_args()
    cls = MODELS[a.model]
    root = a.out / a.model
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames.json").write_text(
        json.dumps(
            dict(
                shift=cls.shift,
                offset=cls.offset,
                thresholds=list(cls.thresholds),
                medians=list(cls.medians),
            )
        )
        + "\n"
    )
    items = []
    for s in map(Path, a.sets):
        n = {k: int(v) for k, v in table(s / "utt2num_samples").items()}
        items += [(n[r], s, r, e.strip()) for r, e in table(s / "wav.scp").items()]
    items.sort(key=lambda x: (-x[0], str(x[1]), x[2]))
    mine = items[a.task - 1 :: a.ntasks]
    model = cls(model_dir=a.model_dir)
    scps, t0, seconds = {}, time.perf_counter(), 0.0
    for n, s, rid, entry in mine:
        x = load(entry)
        prob = np.asarray(model(x), dtype=np.float32)
        od = root / "diar_eval" / f"diarized_{s.name}"
        (od / "predictions").mkdir(parents=True, exist_ok=True)
        path = od / "predictions" / f"{rid}.npy"
        np.save(path, prob[:, None])
        scps.setdefault(od, []).append(f"{rid} {path.resolve()}\n")
        seconds += len(x) / FS
    for od, lines in scps.items():
        (od / f"diarize.{a.task}.scp").write_text("".join(sorted(lines)))
    el = time.perf_counter() - t0
    print(
        f"{a.model} task {a.task}/{a.ntasks}: {len(mine)} recordings, "
        f"{seconds / 3600:.2f} h in {el:.0f} s (RTF {el / max(seconds, 1e-9):.4f})"
    )


if __name__ == "__main__":
    main()

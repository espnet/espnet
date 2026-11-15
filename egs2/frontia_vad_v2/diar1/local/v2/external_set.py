#!/usr/bin/env python3
"""A Kaldi-style test set of another corpus as a VAD test set of this recipe.

    python3 local/v2/external_set.py <kaldi dir> <out dir> \
        [--wav_dir <dir>] [--uem <file>]

<kaldi dir> holds wav.scp (recording id and an audio file path), rttm (any number of
speakers) and optionally a UEM (all.uem or uem, or --uem). Writes <out dir>:
- wav.scp: the audio paths; with --wav_dir, each file is taken from that directory by
  its file name (for a list written on another machine);
- utt2num_samples: the length at 16 kHz;
- rttm: the union of all speakers' segments as one pseudo speaker "speech" (VAD
  reference), md-eval format;
- uem: the UEM lines of the listed recordings (score_vad.py score --uem).
"""

import argparse
import collections
import math
import sys
from pathlib import Path

FS = 16000


def merge(segs):
    out = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--wav_dir", type=Path, default=None)
    ap.add_argument("--uem", type=Path, default=None)
    a = ap.parse_args()
    import soundfile

    wavs = {}
    for line in (a.data / "wav.scp").read_text().splitlines():
        if not line.strip():
            continue
        rid, path = line.split(maxsplit=1)
        path = path.strip()
        if path.endswith("|"):
            sys.exit(f"{a.data}/wav.scp: {rid} is a command, not a file")
        if a.wav_dir:
            path = str(a.wav_dir / Path(path).name)
        wavs[rid] = path
    segs = collections.defaultdict(list)
    for line in (a.data / "rttm").read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER" and p[1] in wavs:
            s = float(p[3])
            segs[p[1]].append((s, s + float(p[4])))
    a.out.mkdir(parents=True, exist_ok=True)
    nums, missing = {}, []
    for rid, path in sorted(wavs.items()):
        try:
            info = soundfile.info(path)
        except Exception as e:  # noqa: BLE001 - report every unreadable file
            missing.append(f"{rid} {path}: {e}")
            continue
        nums[rid] = math.ceil(info.frames * FS / info.samplerate)
    if missing:
        sys.exit(f"{len(missing)} unreadable audio files, e.g. {missing[:3]}")
    (a.out / "wav.scp").write_text("".join(f"{r} {wavs[r]}\n" for r in sorted(nums)))
    (a.out / "utt2num_samples").write_text(
        "".join(f"{r} {n}\n" for r, n in sorted(nums.items()))
    )
    rttm = [
        f"SPEAKER {r} 1 {s:.3f} {e - s:.3f} <NA> <NA> speech <NA> <NA>\n"
        for r in sorted(nums)
        for s, e in merge(segs.get(r, []))
    ]
    (a.out / "rttm").write_text("".join(rttm))
    uem = a.uem or next(
        (a.data / n for n in ("all.uem", "uem") if (a.data / n).exists()), None
    )
    if uem:
        lines = [
            x
            for x in uem.read_text().splitlines()
            if x.split()[:1] and x.split()[0] in nums
        ]
        (a.out / "uem").write_text("".join(x + "\n" for x in lines))
    speech = sum(e - s for r in nums for s, e in merge(segs.get(r, [])))
    print(
        f"{a.out}: {len(nums)} recordings, {sum(nums.values()) / FS / 3600:.2f} h, "
        f"reference speech {speech / 3600:.2f} h, "
        f"recordings without reference speech {sum(1 for r in nums if not segs.get(r))}"
        + (f", UEM {uem}" if uem else ", no UEM (whole recordings)")
    )


if __name__ == "__main__":
    main()

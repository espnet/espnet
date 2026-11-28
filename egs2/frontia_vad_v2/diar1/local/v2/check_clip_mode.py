#!/usr/bin/env python3
"""Check stage 4's clip mode on real packs: a clip labelled inside a pack must
get the labels it gets when it is a recording of its own.

    python3 local/v2/check_clip_mode.py --librispeech <LibriSpeech dir> \
        --out exp/clip_mode_check [--subset dev-clean] [--utterances 60] \
        -- <align.py options, e.g. --word_timing tokens --pad_start 0.05 ...>

The first --utterances utterances of the subset are written twice with
vad_dump.VadDumpWriter (label source A): packed (--pack_seconds, as stage 3
packs short recordings) and one recording per utterance. Both are aligned by
align.py --clip_mode with the given options; then every utterance's speech
frames (10 ms) inside its pack, shifted by its offset, are compared with its
own recording's. Prints the frames that differ (total and the largest per
clip) and the recordings either run dropped; exits 1 when a clip differs by
more than --tolerance frames at any boundary.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import read_clips  # noqa: E402
from vad_dump import VadDumpWriter  # noqa: E402

FRAME = 0.01


def utterances(root: Path, subset: str, n: int):
    for trans in sorted((root / subset).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            yield uid, trans.parent / f"{uid}.flac", text
            n -= 1
            if n == 0:
                return


def write(out: Path, utts, pack_seconds):
    with VadDumpWriter(
        out, "ls", "read", "A", "CC BY 4.0", pack_seconds=pack_seconds
    ) as w:
        for uid, path, text in utts:
            x, rate = soundfile.read(str(path), dtype="float32")
            w.add(uid, x, rate, text=text, lang_sym="<eng>", lang="eng")


def speech(dump: Path):
    """{rec_id: [(start s, end s)]} of a dump's espnet_rttm (sample units)."""
    out = {}
    for line in (dump / "espnet_rttm").open():
        p = line.split()
        if p[0] == "SPEAKER":
            out.setdefault(p[1], []).append((int(p[3]) / 16000, int(p[4]) / 16000))
    return out


def mask(spans, start, end):
    n = int(round((end - start) / FRAME))
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        a, b = max(s, start), min(e, end)
        if b > a:
            m[int(round((a - start) / FRAME)) : int(round((b - start) / FRAME))] = True
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--librispeech", required=True, type=Path)
    ap.add_argument("--subset", default="dev-clean")
    ap.add_argument("--utterances", type=int, default=60)
    ap.add_argument("--pack_seconds", type=float, default=30.0)
    ap.add_argument("--tolerance", type=int, default=1, help="frames per boundary")
    ap.add_argument("--out", required=True, type=Path)
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1 :] if "--" in argv else []
    a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    utts = list(utterances(a.librispeech, a.subset, a.utterances))
    packed, single = a.out / "packed", a.out / "single"
    write(packed, utts, a.pack_seconds)
    write(single, utts, None)
    for d in (packed, single):
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "align.py"),
            "--dump",
            str(d),
        ]
        subprocess.run(cmd + ["--clip_mode", "true"] + rest, check=True)
    clips = read_clips(packed / "before_align" / "clips")
    sp_pack, sp_single = speech(packed), speech(single)
    dropped_pack = {
        x.split("\t")[0]
        for x in (packed / "align_dropped.tsv").read_text().splitlines()
    }
    dropped_single = {
        x.split("\t")[0]
        for x in (single / "align_dropped.tsv").read_text().splitlines()
    }
    total, worst, checked, bad = 0, 0, 0, []
    for rid, cs in clips.items():
        if rid in dropped_pack:
            continue
        for cid, s, e, _ in cs:
            if cid in dropped_single:
                bad.append(f"{cid}: kept in the pack, dropped alone")
                continue
            start, end = s / 16000, e / 16000
            m_pack = mask(sp_pack.get(rid, []), start, end)
            m_one = mask(sp_single.get(cid, []), 0.0, end - start)
            n = min(len(m_pack), len(m_one))
            diff = int((m_pack[:n] != m_one[:n]).sum())
            # boundaries of this clip's speech: 2 per segment
            bounds = 2 * max(1, len(sp_single.get(cid, [])))
            total += diff
            worst = max(worst, diff)
            checked += 1
            if diff > a.tolerance * bounds:
                bad.append(f"{cid}: {diff} frames differ")
    packs = sum(1 for r in clips if len(clips[r]) > 1)
    print(
        f"{len(utts)} utterances in {len(clips)} recordings ({packs} packs of several "
        f"clips); dropped: {len(dropped_pack)} packed recordings, "
        f"{len(dropped_single)} single; {checked} clips compared, {total} frames "
        f"differ in all (largest per clip {worst})"
    )
    for b in bad:
        print("DIFFERS", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

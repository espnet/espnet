#!/usr/bin/env python3
"""Write the labels of aligned dumps again with another padding and min_pause.

    python3 local/v2/relabel.py --word_timing tokens --pad_start 0.05 \
        --pad_end 0.4 --min_pause 0.5 dump/v2_1/<corpus>/<split>[/shard<k>] ...

For a dump aligned by align.py --clip_mode (align.tsv with the clip column,
the file clips), the speech of every kept recording is computed again from
its word spans as align.py --clip_mode does (align.pack_speech: words joined
within a clip only, padding clipped to the clip's edges), without aligning
again. Rewrites espnet_rttm and the speech_seconds column of manifest.tsv,
and records the settings in align_config.json under "labels". The recordings
kept or dropped do not change: the word scores do not depend on the padding.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import pack_speech, read_clips, read_table  # noqa: E402
from vad_dump import MANIFEST_FIELDS  # noqa: E402

COLUMNS = {"segmentation": (5, 6), "tokens": (7, 8)}


def recording_speech(words, clips, fs, min_pause, pad_start, pad_end):
    """[(start s, end s)] of one recording from its word rows and clips.

    words: [(clip index, start s, end s)] in recording time; clips: the
    recording's [(clip id, first sample, end sample, text)].
    """
    per_clip = [[] for _ in clips]
    for k, s, e in words:
        per_clip[k].append((s - clips[k][1] / fs, e - clips[k][1] / fs))
    return pack_speech(
        [(c[1] / fs, c[2] / fs, w) for c, w in zip(clips, per_clip)],
        min_pause,
        pad_start,
        pad_end,
    )


def rttm_lines(rid, segments, n, fs):
    lines, total = [], 0
    for s, e in segments:
        s, e = max(0, int(round(s * fs))), min(n, int(round(e * fs)))
        if e > s:
            lines.append(f"SPEAKER {rid} 1 {s} {e} <NA> <NA> {rid} <NA>\n")
            total += e - s
    lines.append(f"END {rid} <NA> <NA> {n} <NA> <NA> <NA> <NA>\n")
    return lines, total


def relabel(d: Path, timing, min_pause, pad_start, pad_end, fs=16000):
    config = json.loads((d / "align_config.json").read_text())
    if not config.get("clip_mode"):
        raise ValueError(f"{d}: not aligned with --clip_mode")
    a, b = COLUMNS[timing]
    kept = read_table(d / "wav.scp").keys()
    num_samples = {k: int(v) for k, v in read_table(d / "utt2num_samples").items()}
    clips = read_clips(d / "clips")
    words = {}
    for line in (d / "align.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] in kept:
            words.setdefault(f[0], []).append((int(f[9]), float(f[a]), float(f[b])))
    rttm, seconds = [], {}
    for rid in sorted(kept):
        segs = recording_speech(
            words.get(rid, []), clips[rid], fs, min_pause, pad_start, pad_end
        )
        lines, total = rttm_lines(rid, segs, num_samples[rid], fs)
        rttm += lines
        seconds[rid] = total / fs
    (d / "espnet_rttm").write_text("".join(sorted(rttm)))
    with (d / "manifest.tsv").open() as f:
        cols = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(cols, x.rstrip("\n").split("\t"))) for x in f]
    with (d / "manifest.tsv").open("w") as g:
        g.write("\t".join(MANIFEST_FIELDS) + "\n")
        for row in rows:
            row["speech_seconds"] = f"{seconds[row['rec_id']]:.3f}"
            g.write("\t".join(row.get(k, "") for k in MANIFEST_FIELDS) + "\n")
    config["labels"] = dict(
        word_timing=timing, min_pause=min_pause, pad_start=pad_start, pad_end=pad_end
    )
    (d / "align_config.json").write_text(json.dumps(config, indent=1) + "\n")
    return sum(seconds.values()) / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--word_timing", choices=sorted(COLUMNS), default="tokens")
    ap.add_argument("--min_pause", type=float, required=True)
    ap.add_argument("--pad_start", type=float, required=True)
    ap.add_argument("--pad_end", type=float, required=True)
    a = ap.parse_args()
    for d in a.dumps:
        h = relabel(d, a.word_timing, a.min_pause, a.pad_start, a.pad_end)
        print(f"{d}: {h:.2f} h of speech")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Index fixed-length windows of every recording of a dumped data directory.

Reads ``wav.scp`` (Omni-IO entries), ``utt2num_samples`` and ``espnet_rttm``
of a dump directory and writes a Kaldi-style data directory with one segment
per window of ``--window`` samples taken every ``--shift`` samples (the tail
shorter than a window is dropped, as the chunk iterator does):

    wav.scp          the recordings, as in the source (read with ``segments``
                     through the ``omniio_segments`` data type)
    segments         <rec>-<offset> <rec> <start s> <end s>
    vad_intervals    <rec>-<offset> <window> <start> <end> ...  (samples, end excl.)
    speech_shape     <rec>-<offset> <window>
    spk_labels_shape <rec>-<offset> <window>,1

Segments are cut with ``espnet2.fileio.segments.seconds_to_sample`` (as
``SegmentsExtractor`` cuts them); the times are written so that it gives
back the exact sample (checked for every window).

Windows without speech are kept (all-zero label). The label of a window is
the recording's label of ``RttmReader`` sliced to the window: a SPEAKER line
``start end`` marks samples start..end inclusive.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from espnet2.fileio.segments import seconds_to_sample


def read_kv(path):
    with open(path, encoding="utf-8") as f:
        return dict(line.rstrip("\n").split(maxsplit=1) for line in f if line.strip())


def merged_speech(rttm):
    """Recording id -> sorted, merged speech intervals [start, end) in samples."""
    turns = defaultdict(list)
    with open(rttm, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if p and p[0] == "SPEAKER":
                turns[p[1]].append((int(float(p[3])), int(float(p[4])) + 1))
    merged = {}
    for rec, segs in turns.items():
        out = []
        for s, e in sorted(segs):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        merged[rec] = out
    return merged


def seconds(sample, fs):
    """Time of `sample` that seconds_to_sample turns back into `sample`."""
    t = repr(sample / fs)
    if seconds_to_sample(float(t), fs) != sample:
        raise ValueError(f"sample {sample} at {fs} Hz has no exact time in seconds")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="dumped data dir (e.g. dump/raw/train)")
    ap.add_argument("dst", type=Path, help="output dir")
    ap.add_argument("--window", type=int, default=160000, help="samples (10 s)")
    ap.add_argument("--shift", type=int, default=160000, help="samples")
    ap.add_argument("--fs", type=int, default=16000, help="sampling rate (Hz)")
    a = ap.parse_args()

    wav = read_kv(a.src / "wav.scp")
    nsamp = {k: int(v) for k, v in read_kv(a.src / "utt2num_samples").items()}
    speech = merged_speech(a.src / "espnet_rttm")
    a.dst.mkdir(parents=True, exist_ok=True)
    with open(a.dst / "wav.scp", "w", encoding="utf-8") as fw:
        for rec in sorted(wav):
            fw.write(f"{rec} {wav[rec]}\n")
    n_win = n_empty = 0
    with (
        open(a.dst / "segments", "w", encoding="utf-8") as fseg,
        open(a.dst / "vad_intervals", "w", encoding="utf-8") as fl,
        open(a.dst / "speech_shape", "w", encoding="utf-8") as fs,
        open(a.dst / "spk_labels_shape", "w", encoding="utf-8") as fsl,
    ):
        for rec in sorted(wav):
            segs, j = speech.get(rec, []), 0
            for off in range(0, nsamp[rec] - a.window + 1, a.shift):
                end = off + a.window
                while j < len(segs) and segs[j][1] <= off:
                    j += 1  # intervals are sorted; skip those ending before
                bounds = []
                k = j
                while k < len(segs) and segs[k][0] < end:
                    bounds += [max(segs[k][0], off) - off, min(segs[k][1], end) - off]
                    k += 1
                wid = f"{rec}-{off:010d}"
                fseg.write(f"{wid} {rec} {seconds(off, a.fs)} {seconds(end, a.fs)}\n")
                fl.write(" ".join(map(str, [wid, a.window] + bounds)) + "\n")
                fs.write(f"{wid} {a.window}\n")
                fsl.write(f"{wid} {a.window},1\n")
                n_win += 1
                n_empty += not bounds
    print(
        f"{a.src} -> {a.dst}: {len(wav)} recordings, {n_win} windows "
        f"({n_empty} without speech)"
    )


if __name__ == "__main__":
    main()

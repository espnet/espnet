#!/usr/bin/env python3
"""Fixed-length training windows of the recordings of a dumped diar1 data directory.

    make_windows.py SRC DST --window N [--shift N] [--fs 16000]

Reads ``wav.scp``, ``utt2num_samples`` and ``espnet_rttm`` of SRC (a dump of
diar.sh stage 3) and writes DST with one Kaldi segment per window of
``--window`` samples taken every ``--shift`` samples. The tail of a recording
shorter than a window is dropped, as the chunk iterator drops it.

    wav.scp          the recordings of SRC (read through the segments)
    segments         <rec>-<offset> <rec> <start s> <end s>
    espnet_rttm      the labels of each window: SPEAKER lines in samples from
                     the window start (end inclusive, as in SRC) and an END line
                     with the window length; a window without speech has only
                     its END line

The label of a window is the recording's label (``RttmReader``) cut to the
window. A window holds the speakers that speak in it, so its label can have
fewer columns than the model has outputs; the diarization task pads the label
with zeros up to ``num_spk``. The segment times are written so that
``espnet2.fileio.segments.seconds_to_sample`` gives back the exact start and
end samples (checked for every window).
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from espnet2.fileio.segments import seconds_to_sample


def read_kv(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return dict(line.rstrip("\n").split(maxsplit=1) for line in f if line.strip())


def read_turns(rttm: Path) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    """Recording -> speaker -> merged turns [start, end) in samples, sorted.

    Speakers are kept in the order of their first SPEAKER line.
    """
    turns: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(dict)
    with rttm.open(encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if p and p[0] == "SPEAKER":
                start, end = int(float(p[3])), int(float(p[4])) + 1
                turns[p[1]].setdefault(p[7], []).append((start, end))
    merged = {}
    for rec, spks in turns.items():
        merged[rec] = {}
        for spk, segs in spks.items():
            out: List[List[int]] = []
            for s, e in sorted(segs):
                if out and s <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], e)
                else:
                    out.append([s, e])
            merged[rec][spk] = [(s, e) for s, e in out]
    return merged


def window_turns(
    segs: List[Tuple[int, int]], off: int, end: int
) -> List[Tuple[int, int]]:
    """Parts of the sorted, merged turns `segs` inside [off, end), from off."""
    out = []
    for s, e in segs:
        if e <= off:
            continue
        if s >= end:
            break
        out.append((max(s, off) - off, min(e, end) - off))
    return out


def seconds(sample: int, fs: int) -> str:
    """Time of `sample` that seconds_to_sample turns back into `sample`."""
    t = repr(sample / fs)
    if seconds_to_sample(float(t), fs) != sample:
        raise ValueError(f"sample {sample} at {fs} Hz has no exact time in seconds")
    return t


def make_windows(
    src: Path, dst: Path, window: int, shift: int, fs: int
) -> Tuple[int, int, int]:
    """Write DST; return (recordings, windows, windows without speech)."""
    if window <= 0 or shift <= 0:
        raise ValueError(f"window and shift must be positive: {window}, {shift}")
    wav = read_kv(src / "wav.scp")
    nsamp = {k: int(v) for k, v in read_kv(src / "utt2num_samples").items()}
    turns = read_turns(src / "espnet_rttm")
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "wav.scp").open("w", encoding="utf-8") as f:
        for rec in sorted(wav):
            f.write(f"{rec} {wav[rec]}\n")
    n_win = n_empty = 0
    with (
        (dst / "segments").open("w", encoding="utf-8") as fseg,
        (dst / "espnet_rttm").open("w", encoding="utf-8") as frttm,
    ):
        for rec in sorted(wav):
            spks = turns.get(rec, {})
            for off in range(0, nsamp[rec] - window + 1, shift):
                end = off + window
                wid = f"{rec}-{off:010d}"
                fseg.write(f"{wid} {rec} {seconds(off, fs)} {seconds(end, fs)}\n")
                empty = True
                for spk, segs in spks.items():
                    for s, e in window_turns(segs, off, end):
                        frttm.write(
                            f"SPEAKER {wid} 1 {s} {e - 1} <NA> <NA> {spk} <NA>\n"
                        )
                        empty = False
                frttm.write(f"END {wid} <NA> <NA> {window} <NA> <NA> <NA> <NA>\n")
                n_win += 1
                n_empty += empty
    return len(wav), n_win, n_empty


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path, help="dumped data dir (e.g. dump/raw/train)")
    ap.add_argument("dst", type=Path, help="output dir")
    ap.add_argument("--window", type=int, required=True, help="window (samples)")
    ap.add_argument("--shift", type=int, default=None, help="default: --window")
    ap.add_argument("--fs", type=int, default=16000, help="sampling rate (Hz)")
    a = ap.parse_args()
    n_rec, n_win, n_empty = make_windows(
        a.src, a.dst, a.window, a.shift or a.window, a.fs
    )
    print(
        f"{a.src} -> {a.dst}: {n_rec} recordings, {n_win} windows "
        f"({n_empty} without speech)"
    )


if __name__ == "__main__":
    main()

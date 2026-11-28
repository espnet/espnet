"""Harvest an in-domain noise pool from the speech-free gaps of recordings.

DNF and MixIT need noise-only signals. Without an external noise corpus they
can come from the target recordings themselves: every region a VAD calls
non-speech, shrunk by a safety margin, becomes a pool row. Speech activity
comes from labels (RTTM or Kaldi ``segments``) or, when there are none, from
a simple energy VAD.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from espnet3.systems.snoar.src.data.manifest import read_audio

logger = logging.getLogger(__name__)

Intervals = List[Tuple[float, float]]


def read_rttm(path: str | Path) -> Dict[str, Intervals]:
    """Read speech intervals (seconds) per recording from an RTTM file."""
    speech: Dict[str, Intervals] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start, dur = float(parts[3]), float(parts[4])
                speech[parts[1]].append((start, start + dur))
    if not speech:
        raise ValueError(f"no SPEAKER lines in {path}")
    return dict(speech)


def read_segments(path: str | Path) -> Dict[str, Intervals]:
    """Read speech intervals from a Kaldi ``segments`` file."""
    speech: Dict[str, Intervals] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                speech[parts[1]].append((float(parts[2]), float(parts[3])))
    if not speech:
        raise ValueError(f"no `<seg> <reco> <start> <end>` lines in {path}")
    return dict(speech)


def detect_energy_speech(
    x: np.ndarray,
    fs: int,
    frame: float = 0.025,
    hop: float = 0.010,
    margin_db: float = 6.0,
    floor_percentile: float = 10.0,
    min_speech: float = 0.1,
) -> Intervals:
    """Return speech intervals from frame energy above the noise floor.

    A frame is speech when its energy exceeds the ``floor_percentile``-th
    percentile of the recording's frame energies by ``margin_db``; runs
    shorter than ``min_speech`` seconds are dropped. Deliberately simple:
    every speech decision only shrinks the noise pool, so false alarms cost
    pool size, while misses put speech in the pool, which the margin and the
    minimum gap length guard against.
    """
    flen, fhop = int(frame * fs), int(hop * fs)
    if len(x) < flen:
        return []
    n = 1 + (len(x) - flen) // fhop
    idx = np.arange(flen)[None, :] + fhop * np.arange(n)[:, None]
    e = 10 * np.log10(np.mean(x[idx].astype(np.float64) ** 2, axis=1) + 1e-12)
    active = e > np.percentile(e, floor_percentile) + margin_db
    out, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            t0, t1 = start * hop, i * hop + frame
            if t1 - t0 >= min_speech:
                out.append((t0, t1))
            start = None
    return out


def find_gaps(
    speech: Intervals, duration: float, margin: float = 0.1, min_gap: float = 0.3
) -> Intervals:
    """Return the speech-free gaps of ``[0, duration]``.

    Each speech interval is widened by ``margin`` on both sides, and gaps
    shorter than ``min_gap`` are dropped.

    Examples:
        >>> find_gaps([(1.0, 2.0)], 4.0, margin=0.1, min_gap=0.3)
        [(0.0, 0.9), (2.1, 4.0)]
    """
    widened = sorted(
        (max(0.0, s - margin), min(duration, e + margin)) for s, e in speech
    )
    gaps, t = [], 0.0
    for s, e in widened:
        if s - t >= min_gap:
            gaps.append((round(t, 4), round(s, 4)))
        t = max(t, e)
    if duration - t >= min_gap:
        gaps.append((round(t, 4), round(duration, 4)))
    return gaps


def harvest_noise(
    recordings: Iterable[Dict],
    vad: str,
    fs: int = 16000,
    margin: float = 0.1,
    min_gap: float = 0.3,
) -> List[Dict]:
    """Return noise-pool rows from the speech-free gaps of ``recordings``.

    Args:
        recordings: Recording manifest rows (``id``, ``audio``,
            ``num_samples``).
        vad: ``energy``, or the path of an RTTM file (``*.rttm``) or Kaldi
            ``segments`` file whose recording ids match the manifest ids.
        fs: Sampling rate.
        margin: Seconds of each gap given up next to speech.
        min_gap: Shortest gap kept, in seconds.

    Returns:
        Pool rows ``{"audio", "start", "end", "group"}`` (samples).

    Raises:
        ValueError: If the labels match no recording of the manifest.
    """
    recordings = list(recordings)
    labels: Optional[Dict[str, Intervals]] = None
    if vad != "energy":
        path = Path(vad)
        labels = read_rttm(path) if path.suffix == ".rttm" else read_segments(path)
        ids = {r["id"] for r in recordings}
        if not ids & set(labels):
            raise ValueError(
                f"no recording id of the manifest appears in {vad} "
                f"(e.g. manifest {sorted(ids)[:3]}, labels {sorted(labels)[:3]})"
            )
    rows, total = [], 0
    for r in recordings:
        n = int(r["num_samples"])
        if labels is None:
            speech = detect_energy_speech(read_audio(r["audio"], fs=fs), fs)
        else:
            # A recording absent from the labels is skipped, not taken as all
            # noise: missing labels are far likelier than a silent recording.
            if r["id"] not in labels:
                continue
            speech = labels[r["id"]]
        for t0, t1 in find_gaps(speech, n / fs, margin, min_gap):
            s0, s1 = int(t0 * fs), min(n, int(t1 * fs))
            if s1 > s0:
                rows.append(
                    {"audio": r["audio"], "start": s0, "end": s1, "group": r["id"]}
                )
                total += s1 - s0
    logger.info(
        "harvested %d noise regions, %.2f h, from %d recordings",
        len(rows),
        total / fs / 3600,
        len(recordings),
    )
    return rows

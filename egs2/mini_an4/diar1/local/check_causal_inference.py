#!/usr/bin/env python3
"""Check that a diarization model reads no more future than its look-ahead.

For every recording, runs the model on the whole recording and on its first
half, and compares the predictions of the frames of the first half. With
look-ahead L (encoder frames; 0 for a causal model) the first-half frames except
the last L must be the same: a frame's prediction depends on samples up to L
frames after it only. With L > 0 the last L frames must differ for some
recording, which shows that the model does read ahead.

Usage:
    check_causal_inference.py [--num_spk 1] [--look_ahead L] <diar_exp> <wav_scp>
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile
import torch

from espnet2.bin.diar_inference import DiarizeSpeech


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diar_exp", type=Path)
    ap.add_argument("wav_scp", type=Path)
    ap.add_argument("--num_spk", type=int, default=1)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument(
        "--look_ahead",
        type=int,
        default=None,
        help="look-ahead to set on the encoder (None: keep the trained one)",
    )
    a = ap.parse_args()

    diarize = DiarizeSpeech(
        train_config=a.diar_exp / "config.yaml",
        model_file=a.diar_exp / "valid.acc.best.pth",
        num_spk=a.num_spk,
        look_ahead=a.look_ahead,
    )
    look_ahead = getattr(diarize.diar_model.encoder, "look_ahead", 0)
    read_ahead = 0.0
    worst = 0.0
    n = 0
    for line in a.wav_scp.read_text().splitlines():
        rec, path = line.split(maxsplit=1)
        wav, fs = soundfile.read(path, dtype="float32")
        _, full = diarize(torch.from_numpy(wav)[None])
        _, half = diarize(torch.from_numpy(wav[: len(wav) // 2])[None])
        full, half = full[0], half[0]
        assert look_ahead < half.shape[0] <= full.shape[0], (
            rec,
            half.shape,
            full.shape,
        )
        fixed = half.shape[0] - look_ahead  # frames that read no removed sample
        diff = float(np.abs(full[:fixed] - half[:fixed]).max())
        if diff > a.atol:
            raise SystemExit(
                f"{rec}: the first {fixed} frames change by {diff:.3g} when the "
                f"second half of the recording is removed (look-ahead {look_ahead})"
            )
        worst = max(worst, diff)
        if look_ahead > 0:
            read_ahead = max(
                read_ahead,
                float(np.abs(full[fixed : half.shape[0]] - half[fixed:]).max()),
            )
        n += 1
    if look_ahead > 0 and read_ahead <= a.atol:
        raise SystemExit(
            f"look-ahead {look_ahead}: the last {look_ahead} first-half frames never "
            "change, so the model does not read ahead"
        )
    print(
        f"{n} recordings, look-ahead {look_ahead}: the first-half predictions that "
        f"read no removed sample differ by at most {worst:.3g}"
        + (
            f"; the last {look_ahead} differ by up to {read_ahead:.3g}"
            if look_ahead
            else ""
        )
    )


if __name__ == "__main__":
    main()

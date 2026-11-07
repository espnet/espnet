#!/usr/bin/env python3
"""Check that a causal diarization model does not look at future samples.

For every recording, runs the model on the whole recording and on its first
half, and checks that the predictions for the frames of the first half are the
same: a frame's prediction depends on past samples only.

Usage:
    check_causal_inference.py [--num_spk 1] <diar_exp> <wav_scp>
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
    a = ap.parse_args()

    diarize = DiarizeSpeech(
        train_config=a.diar_exp / "config.yaml",
        model_file=a.diar_exp / "valid.acc.best.pth",
        num_spk=a.num_spk,
    )
    worst = 0.0
    n = 0
    for line in a.wav_scp.read_text().splitlines():
        rec, path = line.split(maxsplit=1)
        wav, fs = soundfile.read(path, dtype="float32")
        _, full = diarize(torch.from_numpy(wav)[None])
        _, half = diarize(torch.from_numpy(wav[: len(wav) // 2])[None])
        full, half = full[0], half[0]
        assert 0 < half.shape[0] <= full.shape[0], (rec, half.shape, full.shape)
        diff = float(np.abs(full[: half.shape[0]] - half).max())
        if diff > a.atol:
            raise SystemExit(
                f"{rec}: the first {half.shape[0]} frames change by {diff:.3g} "
                "when the second half of the recording is removed"
            )
        worst = max(worst, diff)
        n += 1
    print(f"{n} recordings: first-half predictions differ by at most {worst:.3g}")


if __name__ == "__main__":
    main()

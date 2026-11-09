#!/usr/bin/env python3
"""Check that a diarization model reads no more future than its look-ahead.

For every recording, changes the samples from a few cut points on and checks
with espnet2.torch_utils.check_look_ahead that the prediction of frame i changes
only if frame i + L reads a changed sample (L: the encoder's look-ahead in
frames; 0 for a causal model). Frame i reads the STFT window of frame i, so with
look-ahead L it may read samples up to the end of frame i + L's window. Some
frame that may read a changed sample must change in some recording, which shows
that the model does read up to its look-ahead.

Usage:
    check_causal_inference.py [--num_spk 1] [--look_ahead L] <diar_exp> <wav_scp>
"""

import argparse
from pathlib import Path

import soundfile
import torch

from espnet2.bin.diar_inference import DiarizeSpeech
from espnet2.torch_utils.check_look_ahead import check_look_ahead, frame_last_sample


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
    # The frames are those of the frontend's STFT; the encoder keeps the rate.
    stft = diarize.diar_model.frontend.stft
    last_sample = frame_last_sample(
        stft.hop_length, stft.n_fft, look_ahead=look_ahead, center=stft.center
    )

    def predict(speech):  # (1, samples) -> (1, frames, num_spk)
        return diarize(speech)[1]

    unread, read, n = 0.0, 0.0, 0
    for line in a.wav_scp.read_text().splitlines():
        rec, path = line.split(maxsplit=1)
        wav, _ = soundfile.read(path, dtype="float32")
        speech = torch.from_numpy(wav)[None]  # (1, samples)
        cuts = [len(wav) * k // 4 for k in (1, 2, 3)]
        try:
            check = check_look_ahead(
                predict, speech, last_sample, cuts, atol=a.atol, tight=False
            )
        except AssertionError as e:
            raise SystemExit(f"{rec} (look-ahead {look_ahead}): {e}")
        unread, read = max(unread, check.unread), max(read, check.read)
        n += 1
    if read <= a.atol:
        raise SystemExit(
            f"look-ahead {look_ahead}: no frame that may read a changed sample "
            "changes, so the model reads less far ahead than its look-ahead"
        )
    print(
        f"{n} recordings, look-ahead {look_ahead}: the frames that read no changed "
        f"sample differ by at most {unread:.3g}; the first frames that may read "
        f"one differ by up to {read:.3g}"
    )


if __name__ == "__main__":
    main()

"""Inference output helpers for Mini AN4 TTS recipes."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    utt_id = data.get("utt_id", str(idx))
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("TTS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "wav": wav}

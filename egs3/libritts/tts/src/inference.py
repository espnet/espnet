"""Inference output helpers for LJSpeech TTS recipes."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    utt_id = data.get("utt_id", str(idx))
    text = str(data.get("raw_text", ""))
    ref = str(data.get("wav_path", "")) # ground truth wav path
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("TTS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "text": text, "ref": ref, "wav": wav}

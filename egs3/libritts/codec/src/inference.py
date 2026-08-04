"""Inference output helpers for the LibriTTS codec recipe."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    """Format one AudioCoding roundtrip result for SCP/artifact writing.

    Args:
        data: Dataset item (``inference: true`` mode), providing ``utt_id``
            and the ground-truth ``wav_path``.
        model_output: Dict returned by
            ``espnet2.bin.gan_codec_inference.AudioCoding.__call__``,
            containing the resynthesized waveform under ``resyn_audio``.
        idx: Dataset index, used as a fallback identifier.

    Returns:
        Dict with ``utt_id``, the ground-truth wav path under ``ref``, and
        the resynthesized waveform under ``wav`` (1-D float32 numpy array,
        materialized as a WAV file via ``output_artifacts``).
    """
    utt_id = str(data.get("utt_id", idx))
    ref = str(data.get("wav_path", ""))  # ground truth wav path
    wav = model_output.get("resyn_audio")
    if wav is None:
        raise RuntimeError("Codec inference output does not contain 'resyn_audio'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "ref": ref, "wav": wav}

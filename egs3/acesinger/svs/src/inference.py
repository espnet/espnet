"""Inference output helpers for the ACE-Opencpop SVS recipe."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    """Assemble one infer-stage record from a dataset sample and model output.

    Wired in through ``output_fn`` in ``conf/inference.yaml``. ``wav`` is the
    synthesized singing written by the ``wav`` output artifact, ``ref`` the
    reference wav path and ``text`` the phoneme sequence, which is what the
    ``measure`` stage reads.

    Args:
        data: One dataset sample built with ``inference: true``.
        model_output: Mapping returned by ``SingingGenerate``; must contain
            ``wav``.
        idx: Index of the sample, used as the utterance id when the sample
            carries no ``utt_id``.

    Returns:
        Dict with ``utt_id``, ``text``, ``ref`` and a 1-D float32 ``wav``.

    Raises:
        RuntimeError: If *model_output* has no ``wav`` entry.
    """
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("SVS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "text": str(data.get("raw_text", "")),
        "ref": str(data.get("wav_path", "")),
        "wav": np.asarray(wav, dtype=np.float32).reshape(-1),
    }

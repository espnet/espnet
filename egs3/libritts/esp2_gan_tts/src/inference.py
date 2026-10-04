"""Inference output helpers for the LibriTTS TTS recipe."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    """Assemble one infer-stage record from a dataset sample and model output.

    Wired in through ``output_fn`` in ``conf/inference.yaml``. The returned
    keys line up with the ``output_artifacts`` declared there and with the
    inputs the ``measure`` stage hands to VERSA: ``wav`` is the synthesized
    audio to score, ``ref`` the ground-truth wav, ``text`` the transcript.

    Args:
        data: One dataset sample. Read with ``.get()``, so a sample built
            without ``inference: true`` still yields a record, using the
            defaults below.
        model_output: Mapping returned by ``Text2Speech``. Must contain
            ``wav``; a torch tensor is detached and moved to CPU.
        idx: Index of the sample. Used as the utterance id when the sample
            carries no ``utt_id``.

    Returns:
        Dict with ``utt_id``, ``text``, ``ref``, and a 1-D float32 ``wav``
        array.

    Raises:
        RuntimeError: If *model_output* has no ``wav`` entry.

    Examples:
        ```python
        import numpy as np

        build_output(
            {"utt_id": "19_198_000000", "raw_text": "hello", "wav_path": "a.wav"},
            {"wav": np.zeros(2, dtype=np.float32)},
            0,
        )
        # -> {'utt_id': '19_198_000000', 'text': 'hello', 'ref': 'a.wav',
        #     'wav': array([0., 0.], dtype=float32)}
        ```
    """
    utt_id = data.get("utt_id", str(idx))
    text = str(data.get("raw_text", ""))
    ref = str(data.get("wav_path", ""))  # ground truth wav path
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("TTS inference output does not contain 'wav'.")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return {"utt_id": utt_id, "text": text, "ref": ref, "wav": wav}

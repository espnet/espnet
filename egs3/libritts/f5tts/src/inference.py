"""Inference output helpers for the LibriTTS F5-TTS recipe."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    """Build the output dict(s) the ``infer`` stage writes to SCP files.

    The template's ``egs3/TEMPLATE/f5tts/src/inference.py`` plus the two
    columns ``conf/metrics.yaml`` reads: ``ref`` is the wav the speaker
    similarity metric compares against (the prompt when the dataset provides
    one, the SIM-o protocol; else the utterance's own ground-truth wav), and
    ``text`` is the untokenized transcript (``raw_text``) the WER metric
    scores against.

    Called with one dataset item, its model output and its index, or, when
    the inference config sets ``batch_size``, with a list of each, in which
    case one dict per item is returned.

    Args:
        data: One dataset sample, or a list of them. ``raw_text`` is the
            target transcript; ``ref_wav_path`` or ``wav_path`` names the
            speaker-similarity reference; ``utt_id`` is used as the
            utterance id when the sample carries one.
        model_output: Mapping returned by the model, holding ``wav``: an
            :class:`espnet3.api.inference.Audio` from
            ``espnet3.systems.f5tts.inference.Inference``, or an array or
            tensor from ``F5TTSInference``. For a batch, a list of such
            mappings, or one mapping whose ``wav`` is a list.
        idx: Index of the sample, or a list of indices. Used as the
            utterance id when the sample carries no ``utt_id``.

    Returns:
        Dict with ``utt_id``, ``text``, ``ref`` and a 1-D float32 ``wav``
        array, or a list of such dicts for a batch.

    Raises:
        RuntimeError: If *model_output* has no ``wav`` entry.

    Example:
        .. code-block:: python

            import numpy as np

            build_output(
                {"raw_text": "hello", "ref_wav_path": "prompt.wav"},
                {"wav": np.zeros(2, dtype=np.float32)},
                7,
            )
            # -> {'utt_id': '7', 'text': 'hello', 'ref': 'prompt.wav',
            #     'wav': array([0., 0.], dtype=float32)}
    """
    if isinstance(data, list):
        if isinstance(model_output, list):
            model_outputs = model_output
        else:
            model_outputs = [{"wav": wav} for wav in model_output["wav"]]
        return [
            build_output(one_data, one_output, one_idx)
            for one_data, one_output, one_idx in zip(data, model_outputs, idx)
        ]
    wav = model_output.get("wav")
    if wav is None:
        raise RuntimeError("F5-TTS inference output does not contain 'wav'.")
    # `Inference` returns an Audio (samples plus rate); keep the samples.
    wav = getattr(wav, "array", wav)
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "text": str(data.get("raw_text", "")),
        "ref": str(data.get("ref_wav_path") or data.get("wav_path", "")),
        "wav": np.asarray(wav, dtype=np.float32).reshape(-1),
    }

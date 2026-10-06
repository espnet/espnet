"""Inference output helpers for F5-TTS recipes."""

from __future__ import annotations

import numpy as np


def build_output(data, model_output, idx):
    """Build the output dict(s) the ``infer`` stage writes to SCP files.

    Wired in through ``output_fn`` in ``conf/inference.yaml``. The returned
    keys line up with ``output_artifacts`` there: ``wav`` is written as an
    audio file and its path stored in ``wav.scp``; ``text`` goes to
    ``text.scp`` as is. A recipe copies this function and adds the columns
    its metrics need, for example the reference wav path for speaker
    similarity.

    Called with one dataset item, its model output and its index, or, when
    the inference config sets ``batch_size``, with a list of each, in which
    case one dict per item is returned.

    Args:
        data: One dataset sample, or a list of them. ``text`` is the target
            text; ``utt_id`` is used as the utterance id when the sample
            carries one.
        model_output: Mapping returned by the model, holding ``wav``: an
            :class:`espnet3.api.inference.Audio` from
            ``espnet3.systems.f5tts.inference.Inference``, or an array or
            tensor from ``F5TTSInference``. For a batch, a list of such
            mappings, or one mapping whose ``wav`` is a list.
        idx: Index of the sample, or a list of indices. Used as the
            utterance id when the sample carries no ``utt_id``.

    Returns:
        Dict with ``utt_id``, ``text`` and a 1-D float32 ``wav`` array, or a
        list of such dicts for a batch.

    Raises:
        RuntimeError: If *model_output* has no ``wav`` entry.

    Example:
        .. code-block:: python

            import numpy as np

            build_output(
                {"text": "hello"}, {"wav": np.zeros(2, dtype=np.float32)}, 7
            )
            # -> {'utt_id': '7', 'text': 'hello',
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
        "text": str(data.get("text", "")),
        "wav": np.asarray(wav, dtype=np.float32).reshape(-1),
    }

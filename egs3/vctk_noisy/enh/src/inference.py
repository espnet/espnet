# egs3/vctk_noisy/enh/src/inference.py
"""Inference output helpers for VCTK-Noisy enhancement recipes."""

import numpy as np


def build_output(data, model_output, idx):
    """Build reference/enhanced output dictionaries for artifact writers.

    Args:
        data: One dataset sample, or a list of samples for batched inference.
        model_output: Speaker-wise model outputs. Each item has shape ``(B, T)``.
        idx: One dataset index, or a list of indices for batched inference.

    Returns:
        One output dictionary for single inference, or one dictionary per item
        for batched inference. Each dictionary contains ``utt_id``, the clean
        ``reference`` waveform, and the first-speaker ``enhanced`` waveform.
    """
    is_batched = isinstance(idx, list)

    if not is_batched:
        utt_id = data.get("utt_id", str(idx))
        reference = np.asarray(data["speech_ref1"], dtype=np.float32)
        enhanced = np.asarray(model_output[0][0], dtype=np.float32)
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        return {"utt_id": utt_id, "reference": reference, "enhanced": enhanced}

    # Batched: return list of dicts
    results = []
    batch_size = len(idx)
    for i in range(batch_size):
        utt_id = data[i].get("utt_id", str(idx[i]))
        reference = np.asarray(data[i]["speech_ref1"], dtype=np.float32)
        enhanced = np.asarray(model_output[0][i], dtype=np.float32)
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        results.append(
            {"utt_id": utt_id, "reference": reference, "enhanced": enhanced}
        )
    return results


class SeparateSpeechWrapper:
    """Adapt ``SeparateSpeech`` to ESPnet3 single and batched inference.

    Args:
        train_config: Enhancement training configuration path.
        model_file: Trained enhancement checkpoint path.
        **kwargs: Additional keyword arguments forwarded to ``SeparateSpeech``.

    Example:
        >>> separator = SeparateSpeechWrapper(
        ...     train_config="exp/train/config.yaml",
        ...     model_file="exp/train/last.ckpt",
        ...     normalize_output_wav=True,
        ... )
    """

    def __init__(self, train_config, model_file, **kwargs):
        from espnet2.bin.enh_inference import SeparateSpeech

        self._model = SeparateSpeech(
            train_config=train_config,
            model_file=model_file,
            **kwargs,
        )

    def __call__(self, speech_mix):
        """Enhance a single waveform or a variable-length waveform batch.

        Args:
            speech_mix: A waveform with shape ``(T,)`` or ``(B, T)``, or a list
                of one-dimensional NumPy waveforms. List inputs may have
                different lengths and are zero-padded to the longest waveform.

        Returns:
            A list with one item per separated speaker. Each item is a tensor
            or NumPy array with shape ``(B, T)``.

        Examples:
            Single inference:

            >>> waveform = np.zeros(16000, dtype=np.float32)
            >>> enhanced = separator(waveform)
            >>> enhanced[0].shape[0]
            1

            Batched inference:

            >>> waveforms = [
            ...     np.zeros(16000, dtype=np.float32),
            ...     np.zeros(12000, dtype=np.float32),
            ... ]
            >>> enhanced = separator(waveforms)
            >>> enhanced[0].shape
            (2, 16000)
        """

        if isinstance(speech_mix, list):
            # Batched: pad to same length and stack to (batch, T)
            max_len = max(len(s) for s in speech_mix)
            padded = np.zeros((len(speech_mix), max_len), dtype=np.float32)
            for i, s in enumerate(speech_mix):
                padded[i, : len(s)] = s
            speech_mix = padded
        else:
            # Single sample: (T,) -> (1, T)
            if speech_mix.ndim == 1:
                speech_mix = speech_mix[np.newaxis, :]

        return self._model(speech_mix)

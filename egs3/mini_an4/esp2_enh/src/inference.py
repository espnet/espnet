"""Inference output helpers for mini_an4 enhancement recipes."""

import numpy as np


def build_output(data, model_output, idx):
    """Build reference/enhanced output dictionaries for artifact writers.

    Args:
        data: One dataset sample, or a list of samples for batched inference.
        model_output: Speaker-wise model outputs. For a single utterance each
            item is ``(T,)`` (or ``(1, T)``). For a batch, each item is a list
            of per-utterance ``(T_i,)`` arrays, or a padded ``(B, T)`` array.
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
        enhanced = np.asarray(model_output[0], dtype=np.float32)
        if enhanced.ndim > 1:
            enhanced = enhanced[0]
        enhanced = enhanced[: reference.shape[0]]
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        return {"utt_id": utt_id, "reference": reference, "enhanced": enhanced}

    results = []
    batch_size = len(idx)
    spk0 = model_output[0]
    for i in range(batch_size):
        utt_id = data[i].get("utt_id", str(idx[i]))
        reference = np.asarray(data[i]["speech_ref1"], dtype=np.float32)
        enhanced = np.asarray(spk0[i], dtype=np.float32)
        enhanced = enhanced[: reference.shape[0]]
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        results.append({"utt_id": utt_id, "reference": reference, "enhanced": enhanced})
    return results


class SeparateSpeechWrapper:
    """Adapt ``SeparateSpeech`` to ESPnet3 single and batched inference.

    Args:
        train_config: Enhancement training configuration path.
        model_file: Trained enhancement checkpoint path.
        **kwargs: Additional keyword arguments forwarded to ``SeparateSpeech``.
    """

    def __init__(self, train_config, model_file, **kwargs):
        """Initialize SeparateSpeech from a training config and checkpoint."""
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
                different lengths and are zero-padded only for the model call.

        Returns:
            A list with one item per separated speaker.
            - single ``(T,)`` input -> each speaker is ``(T,)`` (no batch dim)
            - list input -> each speaker is a list of ``(T_i,)`` arrays
            - already-batched ``(B, T)`` input -> each speaker is ``(B, T)``

        Examples:
            >>> wrapper = SeparateSpeechWrapper(
            ...     train_config="exp/training/config.yaml",
            ...     model_file="exp/training/last.ckpt",
            ... )
            >>> # Single audio: one (T,) waveform per speaker
            >>> enhanced = wrapper(np.zeros(16000, dtype=np.float32))
            >>> enhanced[0].shape
            (16000,)
            >>> # Batched audio: variable lengths, each output keeps its own length
            >>> enhanced = wrapper(
            ...     [np.zeros(16000, np.float32), np.zeros(12000, np.float32)]
            ... )
            >>> [wav.shape for wav in enhanced[0]]
            [(16000,), (12000,)]
        """
        was_list = isinstance(speech_mix, list)
        was_1d = (not was_list) and getattr(speech_mix, "ndim", 1) == 1
        orig_lens = None

        if was_list:
            orig_lens = [len(s) for s in speech_mix]
            max_len = max(orig_lens)
            padded = np.zeros((len(speech_mix), max_len), dtype=np.float32)
            for i, waveform in enumerate(speech_mix):
                padded[i, : len(waveform)] = waveform
            speech_mix = padded
        elif was_1d:
            speech_mix = speech_mix[np.newaxis, :]

        outputs = self._model(speech_mix)

        if was_list:
            return [
                [
                    np.asarray(spk_out[i, : orig_lens[i]], dtype=np.float32)
                    for i in range(len(orig_lens))
                ]
                for spk_out in outputs
            ]
        if was_1d:
            return [np.asarray(spk_out[0], dtype=np.float32) for spk_out in outputs]
        return outputs

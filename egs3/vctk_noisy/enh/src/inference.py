# egs3/vctk_noisy/enh/src/inference.py
"""Inference output helpers for VCTK-Noisy enhancement recipes."""

import numpy as np

def build_output(data, model_output, idx):
    """Build output dict(s) from SeparateSpeech model output.

    Handles both single (idx: int) and batched (idx: list) inference.
    model_output: list of length num_spk, each element shape (batch, T).
    """
    import numpy as np

    is_batched = isinstance(idx, list)

    if not is_batched:
        utt_id = data.get("utt_id", str(idx))
        enhanced = np.asarray(model_output[0][0], dtype=np.float32)
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        return {"utt_id": utt_id, "enhanced": enhanced}

    # Batched: return list of dicts
    results = []
    batch_size = len(idx)
    for i in range(batch_size):
        utt_id = data[i].get("utt_id", str(idx[i]))
        enhanced = np.asarray(model_output[0][i], dtype=np.float32)
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.9
        results.append({"utt_id": utt_id, "enhanced": enhanced})
    return results
    
class SeparateSpeechWrapper:
    """Wraps SeparateSpeech to handle 1D or list input from InferenceRunner.

    InferenceRunner passes either:
    - single sample: 1D numpy array of shape (T,)
    - batched:       list of 1D numpy arrays of varying length

    SeparateSpeech expects (batch, T) and asserts speech_mix.dim() > 1.
    """

    def __init__(self, train_config, model_file, **kwargs):
        from espnet2.bin.enh_inference import SeparateSpeech
        self._model = SeparateSpeech(
            train_config=train_config,
            model_file=model_file,
            **kwargs,
        )

    def __call__(self, speech_mix):
        import numpy as np
        import torch

        if isinstance(speech_mix, list):
            # Batched: pad to same length and stack to (batch, T)
            max_len = max(len(s) for s in speech_mix)
            padded = np.zeros((len(speech_mix), max_len), dtype=np.float32)
            for i, s in enumerate(speech_mix):
                padded[i, :len(s)] = s
            speech_mix = padded
        else:
            # Single sample: (T,) -> (1, T)
            if speech_mix.ndim == 1:
                speech_mix = speech_mix[np.newaxis, :]

        return self._model(speech_mix)

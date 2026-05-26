# egs3/vctk_noisy/enh/src/inference.py
"""Inference output helpers for VCTK-Noisy enhancement recipes."""

import numpy as np


def build_output(data, model_output, idx):
    """Build an output dict from SeparateSpeech model output.

    SeparateSpeech.__call__ asserts speech_mix.dim() > 1, so the runner
    passes speech with shape (T,) which must be expanded to (1, T) before
    calling the model. The model is called via input_key='speech', so we
    handle the unsqueeze inside the model wrapper instead.

    SeparateSpeech returns list of length num_spk, each element shape (batch, T).
    For num_spk=1 and batch=1: model_output[0][0] is the enhanced waveform.

    Args:
        data: Dataset sample dict containing at least ``utt_id``.
        model_output: Return value of SeparateSpeech.__call__,
            list[(batch, T)] of length num_spk.
        idx: Sample index used as fallback utterance ID.

    Returns:
        Dict with ``utt_id`` and ``enhanced`` keys.
    """
    utt_id = data.get("utt_id", str(idx))
    enhanced = np.asarray(model_output[0][0], dtype=np.float32)
    max_val = np.max(np.abs(enhanced))
    if max_val > 1.0:
        enhanced = enhanced / max_val * 0.9
    
    return {"utt_id": utt_id, "enhanced": enhanced}

class SeparateSpeechWrapper:
    """Wraps SeparateSpeech to handle 1D input from InferenceRunner.

    InferenceRunner passes dataset[i]['speech'] directly to the model,
    which is a 1D numpy array of shape (T,). SeparateSpeech expects
    (batch, T) and asserts speech_mix.dim() > 1. This wrapper adds the
    batch dimension before calling the model.

    Usage in inference.yaml:
        model:
          _target_: src.inference.SeparateSpeechWrapper
          train_config: ${exp_dir}/config.yaml
          model_file: ${exp_dir}/last.ckpt
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
        # InferenceRunner passes 1D (T,); SeparateSpeech needs (batch, T).
        if speech_mix.ndim == 1:
            speech_mix = speech_mix[np.newaxis, :]
        return self._model(speech_mix)
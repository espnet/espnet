"""Adapt native SeparateSpeech calls without bypassing shared contract checks."""

import numpy as np

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2TSEInferenceRunner(InferenceRunner):
    """Extract one target per item; the output contract supplies its sample rate."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Unwrap checked mono audio and forward its rate to SeparateSpeech."""
        if any(isinstance(value, (list, tuple)) for value in inputs.values()):
            raise ValueError("TSE inference requires batch_size: null")
        if not all(isinstance(value, Audio) for value in inputs.values()):
            raise TypeError("TSE inference requires checked Audio inputs")
        rates = {value.rate for value in inputs.values()}
        if len(rates) != 1:
            raise ValueError("Mixture and enrollment sample rates must match")
        return {
            **{name: value.array for name, value in inputs.items()},
            "fs": rates.pop(),
        }

    @staticmethod
    def format_output(result, *, batched):
        """Select the first target; shared output validation attaches its rate."""
        if batched:
            raise ValueError("TSE inference requires batch_size: null")
        if not isinstance(result, (list, tuple)) or not result:
            raise ValueError("SeparateSpeech must return at least one target waveform")
        wave = result[0]
        if hasattr(wave, "detach"):
            wave = wave.detach().cpu().numpy()
        wave = np.asarray(wave)
        if wave.ndim == 2 and wave.shape[0] == 1:
            wave = wave[0]
        if wave.ndim != 1 or wave.size == 0:
            raise ValueError(f"Expected one nonempty target waveform, got {wave.shape}")
        return {"inf": wave}

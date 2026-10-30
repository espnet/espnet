"""LACE-specific adaptation of the shared inference runner."""

from __future__ import annotations

import numpy as np

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class LACEInferenceRunner(InferenceRunner):
    """Run ``LACEAudioCoding`` under the inference contract.

    The backend is :class:`~espnet3.systems.lace.src.inference.LACEAudioCoding`.
    The contract hands the runner an :class:`Audio` for ``audio``; the
    backend wants the bare waveform. The backend returns ``codes`` and
    ``resyn_audio``; the contract declares one output, ``resyn_audio``, so
    only the resynthesized waveform is kept, as a 1-D float32 array that the
    backend's ``sample_rate`` turns into an :class:`Audio`.

    ``LACEAudioCoding`` encodes one waveform per call, so the recipe runs
    with ``batch_size: null``; a batched call is refused rather than
    mis-decoded.

    Examples:
        >>> import torch
        >>> LACEInferenceRunner.format_output(
        ...     {"codes": torch.zeros(1, 2, 3), "resyn_audio": torch.zeros(4)},
        ...     batched=False,
        ... )["resyn_audio"].shape
        (4,)
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the backend the waveform of the checked ``audio``.

        Args:
            inputs: The backend's keyword arguments; ``audio`` holds the
                :class:`Audio` the contract produced.

        Returns:
            A copy of ``inputs`` whose ``audio`` is the bare 1-D float32
            waveform.

        Raises:
            ValueError: If ``audio`` is a list, that is, the call is batched.

        Examples:
            >>> audio = Audio(np.zeros(240, dtype=np.float32), 24000)
            >>> out = LACEInferenceRunner.prepare_model_inputs({"audio": audio})
            >>> out["audio"].shape
            (240,)
        """
        inputs = dict(inputs)
        audio = inputs["audio"]
        if isinstance(audio, list):
            raise ValueError(
                "LACEAudioCoding encodes one waveform per call; set "
                "`batch_size: null` in the inference config"
            )
        if isinstance(audio, Audio):
            audio = audio.array
        inputs["audio"] = audio
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Keep the resynthesized waveform as the declared ``resyn_audio`` output.

        Args:
            result: What ``LACEAudioCoding`` returned: a mapping holding
                ``resyn_audio`` (and ``codes``).
            batched: Whether the call was a batch.

        Returns:
            ``{"resyn_audio": <1-D float32 array>}``.

        Raises:
            ValueError: If the call was batched.
            TypeError: If ``result`` is not a mapping.
            KeyError: If ``result`` has no ``resyn_audio``.

        Examples:
            >>> import torch
            >>> out = LACEInferenceRunner.format_output(
            ...     {"resyn_audio": torch.zeros(1, 240)}, batched=False
            ... )
            >>> out["resyn_audio"].shape, out["resyn_audio"].dtype
            ((240,), dtype('float32'))
        """
        if batched:
            raise ValueError(
                "LACEAudioCoding encodes one waveform per call; set "
                "`batch_size: null` in the inference config"
            )
        if not isinstance(result, dict):
            raise TypeError(
                f"LACEAudioCoding must return a dict, got {type(result).__name__}"
            )
        if "resyn_audio" not in result:
            raise KeyError("LACEAudioCoding output has no 'resyn_audio'")
        wav = result["resyn_audio"]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return {"resyn_audio": np.asarray(wav, dtype=np.float32).reshape(-1)}

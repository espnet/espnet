"""Codec-specific adaptation of the shared inference runner."""

from __future__ import annotations

import numpy as np

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class CodecInferenceRunner(InferenceRunner):
    """Run ``espnet2.bin.gan_codec_inference.AudioCoding`` under the contract.

    The contract hands the runner an :class:`Audio` for ``speech``; the
    backend takes the bare waveform as ``audio``. The backend returns ``codes`` and
    ``resyn_audio``; the contract declares one output, ``wav``, so only the
    resynthesized waveform is kept, as a 1-D float32 array that the declared
    ``sample_rate`` of ``contract.outputs.wav`` turns into an :class:`Audio`.

    ``AudioCoding`` encodes one waveform per call, so the recipe runs with
    ``batch_size: null``; a batched call is refused rather than mis-decoded.

    Examples:
        >>> import torch
        >>> CodecInferenceRunner.format_output(
        ...     {"codes": torch.zeros(1, 2, 3), "resyn_audio": torch.zeros(4)},
        ...     batched=False,
        ... )["wav"].shape
        (4,)
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the backend the waveform of the checked ``speech`` as ``audio``.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds the
                :class:`Audio` the contract produced.

        Returns:
            A copy of ``inputs`` without ``speech`` and with ``audio``, the
            bare 1-D float32 waveform, which is the keyword
            ``AudioCoding.__call__`` takes.

        Raises:
            ValueError: If ``speech`` is a list, that is, the call is batched.

        Examples:
            >>> speech = Audio(np.zeros(240, dtype=np.float32), 24000)
            >>> out = CodecInferenceRunner.prepare_model_inputs({"speech": speech})
            >>> sorted(out), out["audio"].shape
            (['audio'], (240,))
        """
        inputs = dict(inputs)
        speech = inputs.pop("speech")
        if isinstance(speech, list):
            raise ValueError(
                "AudioCoding encodes one waveform per call; set `batch_size: null` "
                "in the inference config"
            )
        if isinstance(speech, Audio):
            speech = speech.array
        inputs["audio"] = speech
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Keep the resynthesized waveform as the declared ``wav`` output.

        Args:
            result: What ``AudioCoding`` returned: a mapping holding
                ``resyn_audio`` (and ``codes``).
            batched: Whether the call was a batch.

        Returns:
            ``{"wav": <1-D float32 array>}``.

        Raises:
            ValueError: If the call was batched.
            TypeError: If ``result`` is not a mapping.
            KeyError: If ``result`` has no ``resyn_audio``.

        Examples:
            >>> import torch
            >>> out = CodecInferenceRunner.format_output(
            ...     {"resyn_audio": torch.zeros(1, 240)}, batched=False
            ... )
            >>> out["wav"].shape, out["wav"].dtype
            ((240,), dtype('float32'))
        """
        if batched:
            raise ValueError(
                "AudioCoding encodes one waveform per call; set `batch_size: null` "
                "in the inference config"
            )
        if not isinstance(result, dict):
            raise TypeError(
                f"AudioCoding must return a dict, got {type(result).__name__}"
            )
        if "resyn_audio" not in result:
            raise KeyError("AudioCoding output has no 'resyn_audio'")
        wav = result["resyn_audio"]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return {"wav": np.asarray(wav, dtype=np.float32).reshape(-1)}

"""GAN-TTS adaptation of the shared inference runner."""

from __future__ import annotations

import numpy as np

from espnet3.systems.base.inference_runner import InferenceRunner


class TTSInferenceRunner(InferenceRunner):
    """Run ``espnet2.bin.tts_inference.Text2Speech`` under the contract.

    The contract hands the runner the raw ``text`` and, for a multi-speaker
    model, the ``spembs`` vector; ``Text2Speech`` takes both as they are (it
    tokenizes the text with the preprocessing of its own training config,
    ``${exp_dir}/config.yaml``), so the inputs need no adaptation. It returns
    ``wav`` beside ``att_w``, ``feat_gen``, ``duration`` and other
    diagnostics; the contract declares one output, ``wav``, so only the
    waveform is kept, as a 1-D float32 array that the declared
    ``sample_rate`` of ``contract.outputs.wav`` turns into an
    :class:`~espnet3.contract.kinds.audio.Audio`.

    ``Text2Speech`` synthesizes one utterance per call, so the recipe runs
    with ``batch_size: null``; a batched call is refused rather than
    mis-decoded.

    Examples:
        >>> import torch
        >>> TTSInferenceRunner.format_output(
        ...     {"wav": torch.zeros(4), "att_w": None, "duration": torch.zeros(2)},
        ...     batched=False,
        ... )["wav"].shape
        (4,)
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the checked inputs to ``Text2Speech`` unchanged, one item at a time.

        Args:
            inputs: The backend's keyword arguments: ``text`` (a ``str``) and,
                for a multi-speaker model, ``spembs`` (a 1-D array).

        Returns:
            ``inputs`` itself.

        Raises:
            ValueError: If an input is a list, that is, the call is batched.

        Examples:
            >>> TTSInferenceRunner.prepare_model_inputs({"text": "hi"})
            {'text': 'hi'}
        """
        if any(isinstance(value, list) for value in inputs.values()):
            raise ValueError(
                "Text2Speech synthesizes one utterance per call; set "
                "`batch_size: null` in the inference config"
            )
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Keep the synthesized waveform as the declared ``wav`` output.

        Args:
            result: What ``Text2Speech`` returned: a mapping holding ``wav``
                (and its diagnostics).
            batched: Whether the call was a batch.

        Returns:
            ``{"wav": <1-D float32 array>}``.

        Raises:
            ValueError: If the call was batched.
            TypeError: If ``result`` is not a mapping.
            KeyError: If ``result`` has no ``wav``.

        Examples:
            >>> import torch
            >>> out = TTSInferenceRunner.format_output(
            ...     {"wav": torch.zeros(240)}, batched=False
            ... )
            >>> out["wav"].shape, out["wav"].dtype
            ((240,), dtype('float32'))
        """
        if batched:
            raise ValueError(
                "Text2Speech synthesizes one utterance per call; set "
                "`batch_size: null` in the inference config"
            )
        if not isinstance(result, dict):
            raise TypeError(
                f"Text2Speech must return a dict, got {type(result).__name__}"
            )
        if "wav" not in result or result["wav"] is None:
            raise KeyError("Text2Speech output has no 'wav'")
        wav = result["wav"]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return {"wav": np.asarray(wav, dtype=np.float32).reshape(-1)}

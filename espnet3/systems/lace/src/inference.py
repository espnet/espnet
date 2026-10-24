"""The encode -> decode roundtrip of a fine-tuned LACE codec.

:class:`LACEAudioCoding` is the ``model`` of a LACE inference config, the
analog of ``espnet2.bin.gan_codec_inference.AudioCoding`` for models of
:mod:`espnet3.systems.lace.src.lace`. The contract declared in the template's
``conf/inference.yaml`` (one ``audio`` field in, one ``resyn_audio`` field
out, both at the codec's :attr:`~LACEAudioCoding.sample_rate`) is checked
around every call by the shared runner, and
:class:`~espnet3.systems.lace.src.inference_runner.LACEInferenceRunner`
hands this class the bare waveform and keeps the resynthesis it returns.

Example:
    From a bundle, by directory or Hub tag, through the publication loader;
    any ``LACEAudioCoding`` argument replaces the packed one, which is how a
    rate sweep reads:

    .. code-block:: python

        >>> from espnet3.publication import InferenceModel
        >>> model = InferenceModel.from_pretrained(
        ...     "espnet/libritts_lace_train_encodec_lace_libritts", rate=0.25
        ... )
        >>> model("utt.wav")["resyn_audio"].rate
        24000

    In the ``infer`` stage, ``inference.yaml`` names this class with the
    same arguments:

    .. code-block:: yaml

        model:
          _target_: espnet3.systems.lace.src.inference.LACEAudioCoding
          train_config: ${exp_dir}/lace_model.yaml
          model_file: ${exp_dir}/last.ckpt
          rate: 0.5
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

from espnet3.systems.lace.src.lace import LACE, load_model_state_strict


class LACEAudioCoding:
    """Encode -> decode roundtrip for a LACE codec.

    Analog of ``espnet2.bin.gan_codec_inference.AudioCoding`` for models
    of ``espnet3.systems.lace.src.lace.LACE``: the wrapped architecture is
    rebuilt from ``train_config``, the model spec ``LACESystem.train``
    writes during fine-tuning (``${exp_dir}/lace_model.yaml``), and the
    fine-tuned weights are loaded strictly from ``model_file`` (a Lightning
    checkpoint is unwrapped automatically).

    The compression ``rate`` and the optional ``anchor_start_layer``
    (layers at/after it constrain their boundaries to the union of the
    earlier layers' boundaries) are delivered to the quantizer wrapper via
    its ``set_inference_*`` state around the unmodified ``codec.encode``
    call (which forwards no extra kwargs), and can be overridden per call
    via ``decode_conf={"rate": ..., "anchor_start_layer": ...}``.

    Example:
        .. code-block:: python

            >>> coding = LACEAudioCoding(
            ...     train_config="exp/train_encodec_lace_libritts/lace_model.yaml",
            ...     model_file="exp/train_encodec_lace_libritts/last.ckpt",
            ...     rate=0.5,
            ... )
            >>> coding.sample_rate
            24000
    """

    def __init__(
        self,
        train_config: str,
        model_file: str,
        rate: Optional[float] = None,
        anchor_start_layer: Optional[int] = None,
        target_bandwidth: Optional[float] = None,
        dtype: str = "float32",
        device: Union[str, torch.device] = "cpu",
    ):
        """Rebuild the wrapped codec from its spec and load the fine-tuned weights.

        Args:
            train_config: Path to the ``lace_model.yaml`` spec of a
                fine-tuning run.
            model_file: Path to the fine-tuned weights (model-level state dict
                or Lightning checkpoint).
            rate: Default compression rate; ``None`` falls back to the
                compression model's ``eval_rate``.
            anchor_start_layer: Default index of the first anchored layer;
                ``None`` disables anchoring.
            target_bandwidth: Default target bandwidth in kbps; ``None`` uses
                all codebook layers.
            dtype: Torch dtype name of the model and the input audio.
            device: Device the model runs on.

        Raises:
            RuntimeError: If ``model_file`` does not match the rebuilt
                architecture exactly (propagated from the strict load).
        """
        with open(train_config, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        model = LACE(**spec)
        load_model_state_strict(model, model_file)
        model.to(device).to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.quantizer = model.codec.generator.quantizer
        # The rate the codec runs at, named as espnet2's AudioCoding names it.
        self.fs = int(model.codec.fs)
        self.device = device
        self.dtype = dtype
        self.rate = rate
        self.anchor_start_layer = anchor_start_layer
        self.target_bandwidth = target_bandwidth

    @property
    def sample_rate(self) -> int:
        """Return the rate the codec takes and gives, read by the audio contract.

        The ``audio`` kind resamples an input to this rate and stamps a bare
        output array with it when the contract names no ``sample_rate``, so
        one template serves 16 kHz and 24 kHz codecs alike.

        Returns:
            The codec's sampling rate in Hz (``model.codec.fs``).
        """
        return self.fs

    @torch.no_grad()
    def __call__(
        self,
        audio: Union[torch.Tensor, np.ndarray, None] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
        encode_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the codec roundtrip; mirrors ``AudioCoding.__call__``.

        Args:
            audio: Waveform, (T,) for one utterance or (B, 1, T) for a
                batch.
            decode_conf: Per-call overrides of the constructor defaults;
                accepted keys are ``rate``, ``anchor_start_layer`` and
                ``target_bw``. The name mirrors ``AudioCoding``.
            encode_only: When true, skip decoding and return only the codes.

        Returns:
            Dict with ``codes`` (code indices from ``model.encode``) and,
            unless ``encode_only``, ``resyn_audio`` (the resynthesized
            waveform; 1-D when ``audio`` is 1-D).

        Raises:
            AssertionError: If ``audio`` is ``None``.

        Example:
            .. code-block:: python

                >>> coding = LACEAudioCoding(
                ...     train_config="exp/train_encodec_lace_libritts/lace_model.yaml",
                ...     model_file="exp/train_encodec_lace_libritts/last.ckpt",
                ...     rate=0.5,
                ... )
                >>> output = coding(torch.randn(24000))
                >>> sorted(output)
                ['codes', 'resyn_audio']
                >>> output = coding(torch.randn(24000), decode_conf={"rate": 0.25})
        """
        assert audio is not None, "Audio is invalid, input a valid audio."

        config: Dict[str, Any] = {
            "rate": self.rate,
            "anchor_start_layer": self.anchor_start_layer,
            "target_bw": self.target_bandwidth,
        }
        if decode_conf is not None:
            config.update(decode_conf)

        audio = torch.as_tensor(audio, dtype=getattr(torch, self.dtype)).to(self.device)

        self.quantizer.set_inference_rate(config["rate"])
        self.quantizer.set_inference_anchor_start_layer(config["anchor_start_layer"])
        try:
            codes = self.model.encode(audio, target_bw=config["target_bw"])
        finally:
            self.quantizer.reset_inference_rate()
            self.quantizer.reset_inference_anchor_start_layer()

        output_dict = dict(codes=codes)
        if not encode_only:
            if audio.dim() == 1:
                resyn_audio = self.model.decode(codes).view(-1)
            else:
                resyn_audio = self.model.decode(codes)
            output_dict.update(resyn_audio=resyn_audio)
        return output_dict

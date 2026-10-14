"""The LACE system's side of :mod:`espnet3.api.inference`.

The model behind a LACE bundle is :class:`LACEAudioCoding`, the encode ->
decode roundtrip of a fine-tuned LACE codec at a chosen compression rate
(the analog of ``espnet2.bin.gan_codec_inference.AudioCoding``). The
:class:`Inference` adapter is the whole distance between that and the
contract: one ``audio`` field in, one ``resyn_audio`` field out, at the
codec's sample rate. Building the backend, loading a bundle and the rate
come from :class:`espnet3.systems.base.backend_inference.BackendInference`.

Examples:
    From a bundle, by directory or Hub tag::

        >>> from espnet3.systems.lace.inference import Inference
        >>> model = Inference.from_pretrained("exp/train_encodec_lace/model_pack")
        >>> model("utt.wav")["resyn_audio"].rate
        24000

    Or, without naming the system, through the bundle's ``meta.yaml``, and
    with any ``LACEAudioCoding`` argument replacing the packed one, which is
    how a rate sweep reads::

        >>> from espnet3.api.inference import load
        >>> load("espnet/libritts_lace_train_encodec_lace", rate=0.25)("utt.wav")

    In the ``infer`` stage, ``inference.yaml`` names this class with the
    same arguments::

        model:
          _target_: espnet3.systems.lace.inference.Inference
          train_config: ${exp_dir}/lace_model.yaml
          model_file: ${exp_dir}/last.ckpt
          rate: 0.5
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
import yaml

from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference
from espnet3.systems.lace.src.lace import LACE, load_model_state_strict


class LACEAudioCoding:
    """Encode -> decode roundtrip for a LACE codec.

    Analog of ``espnet2.bin.gan_codec_inference.AudioCoding`` for models
    of ``espnet3.systems.lace.src.lace.LACE``, and the backend
    :class:`Inference` wraps: the wrapped architecture is rebuilt from
    ``train_config``, the model spec ``LACESystem.train`` writes during
    fine-tuning (``${exp_dir}/lace_model.yaml``), and the fine-tuned weights
    are loaded strictly from ``model_file`` (a Lightning checkpoint is
    unwrapped automatically).

    The compression ``rate`` and the optional ``anchor_start_layer``
    (layers at/after it constrain their boundaries to the union of the
    earlier layers' boundaries) are delivered to the quantizer wrapper via
    its ``set_inference_*`` state around the unmodified ``codec.encode``
    call (which forwards no extra kwargs), and can be overridden per call
    via ``decode_conf={"rate": ..., "anchor_start_layer": ...}``.

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

    Examples:
        ```python
        coding = LACEAudioCoding(
            train_config="exp/train_encodec_lace_libritts/lace_model.yaml",
            model_file="exp/train_encodec_lace_libritts/last.ckpt",
            rate=0.5,
        )
        coding.fs  # -> 24000
        ```
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
        """Build the wrapped codec and load its weights."""
        with open(train_config, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        model = LACE(**spec)
        load_model_state_strict(model, model_file)
        model.to(device).to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.quantizer = model.codec.generator.quantizer
        # The rate the codec runs at; read by ``BackendInference.sample_rate``.
        self.fs = int(model.codec.fs)
        self.device = device
        self.dtype = dtype
        self.rate = rate
        self.anchor_start_layer = anchor_start_layer
        self.target_bandwidth = target_bandwidth

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

        Examples:
            ```python
            coding = LACEAudioCoding(
                train_config="exp/train_encodec_lace_libritts/lace_model.yaml",
                model_file="exp/train_encodec_lace_libritts/last.ckpt",
                rate=0.5,
            )
            output = coding(torch.randn(24000))
            sorted(output)  # -> ['codes', 'resyn_audio']
            output = coding(torch.randn(24000), decode_conf={"rate": 0.25})
            ```
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


class Inference(BackendInference):
    """Resynthesize audio through a fine-tuned LACE codec.

    Built three ways, all ending in ``self.backend`` (see
    :class:`BackendInference`):

    - ``Inference.from_pretrained(tag_or_dir, device=...)`` from a
      ``pack_model`` bundle or Hub tag;
    - ``Inference(train_config=..., model_file=..., rate=...)`` from
      :class:`LACEAudioCoding`'s own arguments, which is what
      ``inference.yaml`` does;
    - ``Inference(coding)`` around one already built.

    Called with one utterance (a path, a ``(rate, samples)`` pair, an array
    or an :class:`~espnet3.api.inference.Audio`) it returns
    ``{"resyn_audio": Audio}`` at :attr:`sample_rate`; a multichannel
    recording is reduced to its reference channel first, as the codec is
    single-channel.

    Examples:
        >>> model = Inference.from_pretrained("exp/train_encodec_lace/model_pack")
        >>> out = model("utt.wav")
        >>> out["resyn_audio"].rate == model.sample_rate
        True
        >>> model.batch([{"audio": "a.wav"}, {"audio": "b.wav"}])
        [{'resyn_audio': Audio(...)}, {'resyn_audio': Audio(...)}]
    """

    backend_class = "espnet3.systems.lace.inference.LACEAudioCoding"
    inputs = (Field("audio", "audio", "Input Audio"),)
    outputs = (Field("resyn_audio", "audio", "Resynthesized Audio"),)

    def run(self, audio: Audio) -> Mapping[str, Any]:
        """Return the resynthesized waveform.

        Args:
            audio: The utterance, mono, at :attr:`sample_rate`.

        Returns:
            ``{"resyn_audio": Audio}``, the encode -> decode roundtrip at the
            backend's compression rate, at the codec's sample rate.
        """
        resyn_audio = self.backend(audio.array)["resyn_audio"]
        return {"resyn_audio": Audio(resyn_audio.cpu().numpy(), self.sample_rate)}

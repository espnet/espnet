"""FunASR encoder wrapper for ESPnet2.

Wraps pre-trained FunASR encoders (e.g. Paraformer, SenseVoice) so they can be
used as drop-in encoders inside the ESPnet2 training/inference pipeline.

URL: https://github.com/modelscope/FunASR
"""

import copy
import logging
from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder

try:
    from funasr.auto.auto_model import AutoModel

    is_funasr_available = True
except ImportError:
    is_funasr_available = False
    AutoModel = None


class FunASREncoder(AbsEncoder):
    """ESPnet2 encoder wrapper around a FunASR pre-trained model's encoder.

    The wrapper downloads (or loads from a local path) a FunASR model using
    ``funasr.AutoModel.build_model``, extracts the ``encoder`` sub-module and,
    optionally, the ``normalize`` sub-module, and exposes them through the
    standard :class:`AbsEncoder` interface expected by ESPnet2 tasks.

    Supported model hubs:

    * **ModelScope** (``hub="ms"``, default): ``iic/SenseVoiceSmall``,
      ``iic/paraformer-en``, ``iic/paraformer-zh``, …
    * **HuggingFace** (``hub="hf"``): ``FunAudioLLM/SenseVoiceSmall``, …

    Args:
        input_size (int): Dimension of input fbank features (unused; kept for
            compatibility with ESPnet2 task framework which passes this value
            automatically).
        model_name_or_path (str): FunASR model identifier or local directory
            path.  Passed directly to ``funasr.AutoModel.build_model`` as the
            ``model`` keyword argument.
        hub (str): Model hub to download from.  ``"ms"`` (ModelScope, default)
            or ``"hf"`` (HuggingFace).
        use_normalize (bool): If ``True`` and the underlying FunASR model
            contains a ``normalize`` module, apply it to the input features
            before passing them to the encoder.  Useful when the ESPnet2
            pipeline's own normalizer is disabled.  Default: ``True``.
        freeze_encoder (bool): If ``True``, freeze all encoder parameters so
            they are not updated during fine-tuning.  Default: ``False``.
        download_dir (str, optional): Custom directory for downloading model
            files.  When ``None`` FunASR uses its default cache location.
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 80,
        model_name_or_path: str = "iic/SenseVoiceSmall",
        hub: str = "ms",
        use_normalize: bool = True,
        freeze_encoder: bool = False,
        download_dir: Optional[str] = None,
    ):
        if not is_funasr_available:
            raise ImportError(
                "funasr is not properly installed.  "
                "Please install it with: pip install funasr"
            )

        super().__init__()

        logging.info("Loading FunASR model '%s' from hub '%s'", model_name_or_path, hub)

        build_kwargs = dict(model=model_name_or_path, hub=hub, device="cpu")
        if download_dir is not None:
            build_kwargs["model_revision"] = None
            build_kwargs["cache_dir"] = download_dir

        model, _ = AutoModel.build_model(**build_kwargs)
        model.eval()

        if not hasattr(model, "encoder"):
            raise RuntimeError(
                f"FunASR model '{model_name_or_path}' does not expose an "
                "'encoder' attribute.  Only models with a dedicated encoder "
                "sub-module (e.g. Paraformer, SenseVoiceSmall) are supported."
            )

        self.funasr_encoder = copy.deepcopy(model.encoder)
        self.funasr_encoder.train()

        # Optionally keep the normalizer from the FunASR model.
        self.funasr_normalize = None
        if (
            use_normalize
            and hasattr(model, "normalize")
            and model.normalize is not None
        ):
            self.funasr_normalize = copy.deepcopy(model.normalize)
            self.funasr_normalize.train()

        del model

        if freeze_encoder:
            for param in self.funasr_encoder.parameters():
                param.requires_grad = False
            logging.info("FunASR encoder parameters are frozen.")

        # Cache a copy of the original weights for reload_pretrained_parameters.
        self._pretrained_params = copy.deepcopy(self.funasr_encoder.state_dict())

        self._output_size: int = self.funasr_encoder.output_size()

    # ------------------------------------------------------------------
    # AbsEncoder interface
    # ------------------------------------------------------------------

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode input fbank features.

        Args:
            xs_pad: Input tensor of shape ``(B, T, D)`` where *D* is the fbank
                feature dimension.
            ilens: Input lengths tensor of shape ``(B,)``.
            prev_states: Not used; kept for API compatibility.

        Returns:
            A 3-tuple ``(encoder_out, olens, None)``:

            * ``encoder_out`` — encoded representation ``(B, T', D')``
            * ``olens`` — output lengths ``(B,)``
            * ``None`` — placeholder for hidden states
        """
        if self.funasr_normalize is not None:
            xs_pad, ilens = self.funasr_normalize(xs_pad, ilens)

        enc_out = self.funasr_encoder(xs_pad, ilens)

        # Different FunASR encoder types return either a 2-tuple or 3-tuple.
        if isinstance(enc_out, (tuple, list)):
            if len(enc_out) == 2:
                xs_pad, olens = enc_out
            else:
                xs_pad, olens = enc_out[0], enc_out[1]
            # Unwrap intermediate outputs that some encoders wrap in a tuple.
            if isinstance(xs_pad, (tuple, list)):
                xs_pad = xs_pad[0]
        else:
            raise RuntimeError(
                f"Unexpected output type from FunASR encoder: {type(enc_out)}"
            )

        return xs_pad, olens, None

    # ------------------------------------------------------------------
    # Optional helper kept for compatibility with ESPnet2 fine-tuning
    # utilities that call reload_pretrained_parameters() to reset encoder
    # weights after N warm-up steps.
    # ------------------------------------------------------------------

    def reload_pretrained_parameters(self):
        self.funasr_encoder.load_state_dict(self._pretrained_params)
        logging.info("FunASR encoder pretrained parameters reloaded.")

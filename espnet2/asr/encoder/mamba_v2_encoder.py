import logging
from typing import List, Optional, Tuple
import math
from functools import partial

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling1,
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from espnet2.asr.state_spaces.mamba.mamba_v1 import Block
from espnet2.asr.state_spaces.mamba.mamba_v2 import MambaV2
try:
    from espnet2.asr.state_spaces.mamba.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from espnet2.asr.state_spaces.mamba.serialbimamba import SerialBiMambaBlock

class MambaV2EncoderLayer(torch.nn.Module):
    """MambaV2 encoder layer module."""

    def __init__(
        self,
        d_model: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.ssm_cfg = ssm_cfg
        self.residual_in_fp32 = residual_in_fp32

        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mixer_cls = partial(MambaV2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        self.norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.block = Block(
            d_model,
            self.mixer_cls,
            norm_cls=self.norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.block.layer_idx = layer_idx

    def forward(self, hidden_states, residual=None, mask=None, inference_params=None, flip_fn=None):
        """Forward function."""
        hidden_states, residual = self.block(hidden_states, residual, mask, inference_params, flip_fn=flip_fn)

        return hidden_states, residual, mask


class SerialBiMambaV2EncoderLayer(MambaV2EncoderLayer):
    def __init__(
        self,
        d_model: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__(d_model, ssm_cfg, norm_epsilon, rms_norm, residual_in_fp32, fused_add_norm, layer_idx, device, dtype)
        self.block = SerialBiMambaBlock(
            d_model,
            self.mixer_cls,
            norm_cls=self.norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.block.layer_idx = layer_idx

class MambaV2Encoder(AbsEncoder):
    """Mamba encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        num_blocks: int = 12,
        dropout_rate: float = 0.0,
        pos_enc_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        padding_idx: int = -1,
        layer_drop_rate: float = 0.0,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        interctc_layer_idx=None,
        interctc_use_conditioning: bool = False,
        init_rescale: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.num_blocks = num_blocks
        self.initializer_cfg = initializer_cfg
        
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv1d1":
            self.embed = Conv1dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Dropout(pos_enc_dropout_rate) if pos_enc_dropout_rate > 0.0 else nn.Identity(),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc=nn.Identity(),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = torch.nn.Sequential(
                    nn.Identity()
                )
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: MambaV2EncoderLayer(
                output_size,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                layer_idx=lnum,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(output_size) if not rms_norm else RMSNorm(output_size)

        if interctc_layer_idx is None:
            interctc_layer_idx = []
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

        if init_rescale:
            logging.info("Rescaling weights in MambaEncoder")
            self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        self.apply(
            partial(
                self.init_weights,
                n_layer=self.num_blocks,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

    # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
    def init_weights(
        self,
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        max_layer: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        residual = None

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv1dSubsampling1)
            or isinstance(self.embed, Conv1dSubsampling2)
            or isinstance(self.embed, Conv1dSubsampling3)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if max_layer is not None and 0 <= max_layer < len(self.encoders):
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, residual = encoder_layer(xs_pad, residual)
                    if layer_idx >= max_layer:
                        break
            else:
                xs_pad, residual, masks = self.encoders(xs_pad, residual, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, residual = encoder_layer(xs_pad, residual)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            xs_pad = list(xs_pad)
                            xs_pad[0] = xs_pad[0] + self.conditioning_layer(ctc_out)
                            xs_pad = tuple(xs_pad)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if residual is not None:
            xs_pad = xs_pad + residual
        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

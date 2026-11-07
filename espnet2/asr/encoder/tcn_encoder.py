"""Temporal convolutional network (TCN) encoder.

A thin wrapper that exposes the Conv-TasNet TCN in
``espnet2/enh/layers/tcn.py`` through the ``AbsEncoder`` interface, so that
frame-level tasks such as diarization or voice activity detection can use a
light, optionally causal, convolutional encoder.
"""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.enh.layers.tcn import TemporalConvNet
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

# Normalizations that pool statistics over time and therefore see the future.
_NON_CAUSAL_NORMS = ("gLN", "GN")


class TCNEncoder(AbsEncoder):
    """Stack of dilated depthwise-separable 1-D convolutions (Conv-TasNet TCN).

    The frame rate is kept: the output has as many frames as the input.

    With ``causal=True`` every output frame depends only on the current and the
    past input frames. The normalization must then be per frame (``cLN``) or
    use fixed statistics at inference (``BN``); ``gLN`` and ``GN`` pool over
    the whole sequence and are rejected.

    Args:
        input_size: Input feature dimension.
        output_size: Output dimension.
        bottleneck_size: Channels of the bottleneck 1x1 convolution (B).
        hidden_size: Channels inside each convolutional block (H).
        kernel_size: Kernel size of the depthwise convolutions (P).
        num_blocks: Convolutional blocks per repeat, with dilation 1, 2, ...,
            2**(num_blocks - 1) (X).
        num_repeats: Number of repeats (R).
        norm_type: "cLN", "BN", "gLN" or "GN".
        causal: Use left-only padding so no future frame is read.
        pre_output_nonlinear: Nonlinearity before the output projection:
            "linear", "prelu", "relu", "tanh" or "sigmoid".
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 128,
        bottleneck_size: int = 64,
        hidden_size: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 5,
        num_repeats: int = 2,
        norm_type: str = "cLN",
        causal: bool = True,
        pre_output_nonlinear: str = "prelu",
    ):
        super().__init__()
        if causal and norm_type in _NON_CAUSAL_NORMS:
            raise ValueError(
                f"norm_type={norm_type} pools over time and is not causal; "
                "use 'cLN' or 'BN' with causal=True"
            )
        self._output_size = output_size
        self.causal = causal
        self.tcn = TemporalConvNet(
            N=input_size,
            B=bottleneck_size,
            H=hidden_size,
            P=kernel_size,
            X=num_blocks,
            R=num_repeats,
            C=1,
            out_channel=output_size,
            norm_type=norm_type,
            causal=causal,
            pre_mask_nonlinear=pre_output_nonlinear,
            mask_nonlinear="linear",
        )

    def output_size(self) -> int:
        return self._output_size

    @property
    def receptive_field(self) -> int:
        """Number of input frames each output frame can see."""
        return self.tcn.receptive_field

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode a padded batch.

        Args:
            xs_pad: Input features (B, T, input_size).
            ilens: Input lengths (B,).
            prev_states: Not used.

        Returns:
            Encoded features (B, T, output_size), output lengths (B,), None.
        """
        out = self.tcn(xs_pad.transpose(1, 2))  # (B, 1, output_size, T)
        out = out.squeeze(1).transpose(1, 2)  # (B, T, output_size)
        out = out.masked_fill(make_pad_mask(ilens, out, 1), 0.0)
        return out, ilens, None

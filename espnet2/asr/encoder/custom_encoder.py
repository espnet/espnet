"""Customizable encoder definition."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.asr.custom.build_encoder import build_encoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class CustomEncoder(AbsEncoder):
    """Customizable encoder module.

    Args:
        input_size: Input dim
        architecture: Architecture definition, layer by layer
        repeat: Number of types architecture should be repeated
        transformer_concat_after:
        transformer_normalize_before: Whether to use layer_norm before the first block
    """

    def __init__(
        self,
        input_size: int,
        architecture: List[Dict[str, Any]] = None,
        repeat: int = 0,
        transformer_concat_after=True,
        transformer_normalize_before=True,
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()

        assert (
            architecture is not None
        ), f'{"Architecture configuration for custom model is mandatory."}'

        self.embed, self.encoders, self._output_size = build_encoder(
            input_size, architecture, repeat=repeat, padding_idx=padding_idx
        )

        self.concat_after = transformer_concat_after
        self.normalize_before = transformer_normalize_before

        if self.normalize_before:
            self.after_norm = LayerNorm(self._output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        xs_pad, masks = self.encoders(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens, None

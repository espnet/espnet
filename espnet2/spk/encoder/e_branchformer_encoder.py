"""E-Branchformer encoder for speaker embedding extraction."""

import torch
from typeguard import typechecked

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class EBranchformerSpkEncoder(EBranchformerEncoder):
    """E-Branchformer encoder that outputs frame-level speaker features.

    The blocks are those of the ASR E-Branchformer (Kim et al., SLT 2022). With
    ``aggregate_layer``, the outputs of every block are concatenated along the
    feature axis before the final layer norm, as MFA-Conformer does (Zhang et al.,
    Interspeech 2022), so the output size is ``output_size * num_blocks``.

    Args:
        input_size (int): Input feature dimension.
        aggregate_layer (bool): Concatenate the outputs of all blocks instead of
            returning only the last one.
        **kwargs: Passed to :class:`EBranchformerEncoder`. ``interctc_layer_idx``
            is not supported here.
    """

    @typechecked
    def __init__(self, input_size: int, aggregate_layer: bool = True, **kwargs):
        if kwargs.get("interctc_layer_idx"):
            raise ValueError("interctc_layer_idx is not supported for speaker models")
        super().__init__(input_size, **kwargs)
        self.aggregate_layer = aggregate_layer
        if aggregate_layer:
            self._output_size = self._output_size * len(self.encoders)
            self.after_norm = LayerNorm(self._output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Input features (#batch, time, input_size).

        Returns:
            torch.Tensor: Frame-level features (#batch, output_size, time').
        """
        # speaker training crops fixed-length segments, so nothing is padded
        masks = None
        if isinstance(
            self.embed,
            (
                Conv2dSubsampling,
                Conv2dSubsampling1,
                Conv2dSubsampling2,
                Conv2dSubsampling6,
                Conv2dSubsampling8,
            ),
        ):
            short_status, limit_size = check_short_utt(self.embed, x.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {x.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames)",
                    x.size(1),
                    limit_size,
                )
            x, _ = self.embed(x, masks)
        else:
            x = self.embed(x)

        layer_outs = []
        for encoder_layer in self.encoders:
            # A dropped layer passes its input through, so the concatenation
            # keeps one slot per block.
            if not (
                self.training
                and torch.empty(1).uniform_().item() < self.layer_drop_rate
            ):
                x, _ = encoder_layer(x, masks)
            layer_outs.append(x[0] if isinstance(x, tuple) else x)

        if self.aggregate_layer:
            x = torch.cat(layer_outs, dim=-1)
        else:
            x = layer_outs[-1]
        return self.after_norm(x).transpose(1, 2)

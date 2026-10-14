"""Identity (no-op) compression model."""

from typing import Optional, Union

import torch

from espnet3.systems.lace.src.compression import (
    BaseCompressionModel,
    CompressionOutput,
)


class IdentityCompression(BaseCompressionModel):
    """No-op compression: every input frame becomes its own segment.

    Used as a no-compression baseline when evaluating codec reconstruction
    (the codec sees one code per frame, like standard inference, but the
    pipeline still produces a valid CompressionOutput so downstream
    segment-rate stats and decoding code paths work unchanged).

    Forward kwargs (rate, anchor_boundary, ...) are accepted and ignored so
    this is a drop-in replacement in sweeps; the rate has no effect because
    the output is always per-frame segmentation.

    Example:
        .. code-block:: python

            >>> model = IdentityCompression()
            >>> output = model(torch.randn(2, 50, 128), rate=0.3)
            >>> output.expected_length
            tensor([[50.], [50.]])
    """

    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Return the per-frame segmentation of ``x``.

        Args:
            x: (B, T, D) input features.
            rate: Ignored.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            **kwargs: Ignored (e.g. ``anchor_boundary``).

        Returns:
            ``CompressionOutput`` whose ``reconstructed_features`` is ``x``.

        Example:
            .. code-block:: python

                >>> output = IdentityCompression()(torch.randn(2, 50, 128))
                >>> output.segment_idx[0, :4]
                tensor([0, 1, 2, 3])
        """
        return self._build_no_rate_output(x, padding_mask)

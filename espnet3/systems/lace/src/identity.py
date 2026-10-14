"""Identity (no-op) compression model."""

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

    Examples:
        ```python
        output = IdentityCompression()(torch.randn(2, 50, 128), rate=0.3)
        output.expected_length  # -> tensor([[50.], [50.]])
        ```
    """

    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask=None,
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
        """
        return self._no_rate_output(x, padding_mask)

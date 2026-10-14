"""Multi-compression residual vector quantizer wrappers.

Two layers of wrapping:

- ``RVQCompressionWrapper`` wraps espnet2's ``ResidualVectorQuantization``
  (``espnet2/gan_codec/shared/quantizer/modules/core_vq.py``) and inserts a
  compression-model call before every codebook layer.  Its forward/encode/
  decode mirror the original loops exactly, so with the identity compression
  model (or ``rate=None``) it is bit-equivalent to the unwrapped module.
- ``CompressionResidualVectorQuantizer`` replaces espnet2's
  ``ResidualVectorQuantizer`` (``generator.quantizer``) while keeping its
  exact call interface: ``forward(x, sample_rate, bandwidth)``,
  ``encode(x, sample_rate, bandwidth, st)``, ``decode(codes)``.  Because
  every ``espnet2/gan_codec`` RVQ codec (SoundStream, Encodec, DAC,
  FunCodec) calls its quantizer through exactly this interface, swapping
  the module is all that is needed: the codec's own forward/encode/decode
  and losses run unmodified.

Compression-rate delivery (the codecs never pass a ``rate`` kwarg):

- training: sampled internally per forward pass (``random_rate`` strategy in
  [``min_rate``, ``max_rate``]).  SoundStream/Encodec cache generator
  outputs across the G/D turns of one batch during training, so a single
  sample per batch is used consistently by both turns.
- validation: deterministic ``eval_rate`` (no generator caching in eval,
  and comparable epochs need a fixed rate).
- inference: ``set_inference_rate(rate)`` / ``reset_inference_rate()``,
  because ``codec.encode()`` does not forward extra kwargs.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from espnet3.systems.lace.compression import BaseCompressionModel

RATE_STRATEGIES = ("per_quantizer", "per_sample", "none")


class RVQCompressionWrapper(nn.Module):
    """Multi-mode compression wrapper around ``ResidualVectorQuantization``.

    Applies the compression model to the residual before every codebook
    layer, so each layer quantizes segment-averaged features under its own
    (possibly different) compression rate.

    Args:
        original_vq: espnet2 ``ResidualVectorQuantization`` instance (the
            ``.vq`` attribute of a ``ResidualVectorQuantizer``).
        compression_model: Compression model called before every layer.

    Examples:
        Normally built by ``CompressionResidualVectorQuantizer``; direct use
        on the quantizer of the recipe's Encodec (hidden dim 512, 32 layers)
        looks like:
        ```python
        quantizer = model.codec.generator.quantizer
        wrapper = RVQCompressionWrapper(quantizer.vq, IdentityCompression())
        quantized, codes, losses = wrapper(torch.randn(2, 512, 50), rate=0.5)
        codes.shape  # -> torch.Size([32, 2, 50])
        ```
    """

    def __init__(self, original_vq: nn.Module, compression_model: BaseCompressionModel):
        """Initialize the wrapper."""
        super().__init__()
        self.rvq = original_vq  # espnet2 ResidualVectorQuantization
        self.compression_model = compression_model

    @property
    def layers(self):
        """Return the original codebook layers.

        A property, not an attribute, so the modules are not registered
        twice in the state dict.
        """
        return self.rvq.layers

    @property
    def quantizer_dropout(self):
        """Return the wrapped module's quantizer dropout rate (0.0 if unset)."""
        return getattr(self.rvq, "quantizer_dropout", 0.0)

    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        n_q: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ):
        """Quantize ``x``, compressing the residual before every layer.

        Mirrors ``ResidualVectorQuantization.forward`` of espnet2.

        Args:
            x: (B, D, T) encoder output.
            rate: Compression rate: a float, a (B,) tensor with one rate per
                batch item, a (num_layers, B) tensor with one rate per layer
                and batch item, or ``None`` for no compression.
            n_q: Number of codebook layers to use; ``None`` uses all.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: Layers at or after this index may only place
                boundaries inside the union of the earlier layers'
                boundaries.  ``None`` disables anchoring.

        Returns:
            Without quantizer dropout, the tuple ``(quantized, codes,
            commit_losses)``; with quantizer dropout (DAC), the tuple
            ``(quantized, codes, commit_losses, quantization_losses)``.
            ``quantized`` is (B, D, T), ``codes`` is (n_q, B, T) and the
            losses are stacked per layer.
        """
        quantized_out = 0.0
        residual = x
        rate_layer = rate
        anchor_boundary = None

        if not self.quantizer_dropout:
            all_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)

            for idx, layer in enumerate(self.layers[:n_q]):
                if (
                    rate is not None
                    and isinstance(rate, torch.Tensor)
                    and rate.dim() == 2
                ):
                    rate_layer = rate[idx, :]

                use_anchor = (
                    anchor_start_layer is not None and idx >= anchor_start_layer
                )
                compression_output = self.compression_model(
                    residual.transpose(1, 2),
                    rate=rate_layer,
                    padding_mask=padding_mask,
                    anchor_boundary=anchor_boundary if use_anchor else None,
                )
                compressed_residual = (
                    compression_output.reconstructed_features.transpose(1, 2)
                )

                # Accumulate the boundary union from free layers only; it
                # becomes the anchor set for layers >= anchor_start_layer.
                if anchor_start_layer is not None and not use_anchor:
                    anchor_boundary = (
                        torch.logical_or(
                            anchor_boundary, compression_output.boundary_soft.bool()
                        )
                        if anchor_boundary is not None
                        else compression_output.boundary_soft.bool()
                    )

                quantized, indices, loss = layer(compressed_residual)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)
                all_losses.append(loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
            return quantized_out, out_indices, out_losses
        else:
            all_commit_losses = []
            all_quant_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)
            if self.training:
                n_q = torch.ones((x.shape[0],)) * len(self.layers) + 1
                dropout = torch.randint(1, len(self.layers) + 1, (x.shape[0],))
                n_dropout = int(x.shape[0] * self.quantizer_dropout)
                n_q[:n_dropout] = dropout[:n_dropout]
                n_q = n_q.to(x.device)

            for i, layer in enumerate(self.layers):
                if self.training is False and i >= n_q:
                    break
                mask = torch.full((x.shape[0],), fill_value=i, device=x.device) < n_q

                if (
                    rate is not None
                    and isinstance(rate, torch.Tensor)
                    and rate.dim() == 2
                ):
                    rate_layer = rate[i, :]

                use_anchor = anchor_start_layer is not None and i >= anchor_start_layer
                compression_output = self.compression_model(
                    residual.transpose(1, 2),
                    rate=rate_layer,
                    padding_mask=padding_mask,
                    anchor_boundary=anchor_boundary if use_anchor else None,
                )
                compressed_residual = (
                    compression_output.reconstructed_features.transpose(1, 2)
                )

                if anchor_start_layer is not None and not use_anchor:
                    anchor_boundary = (
                        torch.logical_or(
                            anchor_boundary, compression_output.boundary_soft.bool()
                        )
                        if anchor_boundary is not None
                        else compression_output.boundary_soft.bool()
                    )

                quantized, indices, commit_loss, quant_loss = layer(
                    compressed_residual, mask
                )
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized * mask[:, None, None]

                all_indices.append(indices)
                all_commit_losses.append(commit_loss)
                all_quant_losses.append(quant_loss)

            if self.training:
                quantized_out = x + (quantized_out - x).detach()

            out_commit_losses, out_quant_losses, out_indices = map(
                torch.stack, (all_commit_losses, all_quant_losses, all_indices)
            )
            return quantized_out, out_indices, out_commit_losses, out_quant_losses

    def _encode_loop(
        self,
        x: torch.Tensor,
        rate=None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Shared encode loop used by both encode() and encode_with_segments().

        Returns:
            out_indices:   (n_q, B, T') stacked code indices.
            out_quantized: (n_q, B, D, T') pre-quantization residuals per layer.
            segment_indices: list of (B, T') segment index tensors per layer.
        """
        residual = x
        all_indices = []
        all_quantized = []
        segment_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        rate_layer = rate
        anchor_boundary = None

        for idx, layer in enumerate(self.layers[st:n_q]):
            if rate is not None and isinstance(rate, torch.Tensor) and rate.dim() == 2:
                rate_layer = rate[st + idx, :]  # (Q, B) -> (B,)

            layer_idx = st + idx
            use_anchor = (
                anchor_start_layer is not None and layer_idx >= anchor_start_layer
            )
            all_quantized.append(residual)
            compression_output = self.compression_model(
                residual.transpose(1, 2),
                rate=rate_layer,
                padding_mask=padding_mask,
                anchor_boundary=anchor_boundary if use_anchor else None,
            )
            compressed_residual = compression_output.reconstructed_features.transpose(
                1, 2
            )
            segment_indices.append(compression_output.segment_idx)

            if anchor_start_layer is not None and not use_anchor:
                anchor_boundary = (
                    torch.logical_or(
                        anchor_boundary, compression_output.boundary_soft.bool()
                    )
                    if anchor_boundary is not None
                    else compression_output.boundary_soft.bool()
                )

            indices = layer.encode(compressed_residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        return torch.stack(all_indices), torch.stack(all_quantized), segment_indices

    def encode(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode ``x`` to discrete codes.

        Args:
            x: (B, D, T) encoder output.
            rate: Compression rate; see ``forward``.
            n_q: Index of the last codebook layer to use (exclusive);
                ``None`` uses all.
            st: Index of the first codebook layer to use; ``None`` means 0.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: See ``forward``.

        Returns:
            (n_q - st, B, T) code indices.
        """
        out_indices, _, _ = self._encode_loop(
            x,
            rate=rate,
            n_q=n_q,
            st=st,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """Decode code indices to the summed quantized representation.

        Args:
            q_indices: (n_q, B, T) code indices.

        Returns:
            (B, D, T) quantized representation.
        """
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out

    def encode_with_segments(
        self,
        x: torch.Tensor,
        rate: Optional[Union[float, torch.Tensor]] = None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode and also return per-layer residuals and segment indices.

        Args:
            x: (B, D, T) encoder output.
            rate: Compression rate; see ``forward``.
            n_q: Index of the last codebook layer to use (exclusive);
                ``None`` uses all.
            st: Index of the first codebook layer to use; ``None`` means 0.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: See ``forward``.

        Returns:
            Tuple of the (n_q - st, B, T) code indices, the (n_q - st, B, D,
            T) residuals fed to each layer before compression, and a list
            with one (B, T) segment index tensor per layer.
        """
        return self._encode_loop(
            x,
            rate=rate,
            n_q=n_q,
            st=st,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )


class CompressionResidualVectorQuantizer(nn.Module):
    """Drop-in replacement for espnet2's ``ResidualVectorQuantizer``.

    Holds the original quantizer as ``self.rvq`` and swaps its inner
    ``.vq`` for an ``RVQCompressionWrapper``.  The public interface
    (``forward(x, sample_rate, bandwidth)`` / ``encode`` / ``decode`` and
    the bandwidth getters) matches the original exactly, including the
    ``quantizer_dropout``-dependent 4- vs 5-tuple forward return.

    Args:
        original_rvq: espnet2 ``ResidualVectorQuantizer`` instance
            (``generator.quantizer`` of a built codec).
        compression_model: Compression model called before every layer.
        min_rate: Lower end of the training-time rate sampling range.
        max_rate: Upper end of the training-time rate sampling range.
        random_rate: Sampling strategy: ``per_quantizer`` (one rate per
            codebook layer and batch item), ``per_sample`` (one per batch
            item), or ``none`` (no compression during training).
        eval_rate: Deterministic rate used in eval-mode forward passes
            (validation).  ``None`` means no compression at validation.

    Raises:
        ValueError: If ``random_rate`` is not one of ``RATE_STRATEGIES``.
        TypeError: If ``original_rvq`` has no inner ``.vq`` module, i.e. it
            is not an espnet2 ``ResidualVectorQuantizer``.

    Examples:
        ``LACE`` swaps it in for a codec's quantizer:
        ```python
        generator = model.codec.generator
        generator.quantizer = CompressionResidualVectorQuantizer(
            generator.quantizer,
            CosineSimilarityCompression(),
            min_rate=0.1,
            max_rate=1.0,
            random_rate="per_quantizer",
            eval_rate=0.5,
        )
        ```
    """

    def __init__(
        self,
        original_rvq: nn.Module,
        compression_model: BaseCompressionModel,
        min_rate: float = 0.2,
        max_rate: float = 1.0,
        random_rate: str = "per_quantizer",
        eval_rate: Optional[float] = None,
    ):
        """Initialize the quantizer wrapper."""
        super().__init__()
        if random_rate not in RATE_STRATEGIES:
            raise ValueError(
                f"Unknown random_rate strategy '{random_rate}'. "
                f"Choose from: {RATE_STRATEGIES}."
            )
        if not hasattr(original_rvq, "vq"):
            raise TypeError(
                "CompressionResidualVectorQuantizer expects a quantizer with an "
                "inner '.vq' ResidualVectorQuantization module (espnet2 "
                f"ResidualVectorQuantizer); got {type(original_rvq).__name__}. "
                "HiFiCodec's GroupResidualVectorQuantization is not supported."
            )
        self.rvq = original_rvq
        self.rvq.vq = RVQCompressionWrapper(self.rvq.vq, compression_model)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.random_rate = random_rate
        self.eval_rate = eval_rate
        self._inference_rate = None
        self._inference_anchor_start_layer = None

    # ------------------------------------------------------------------
    # Rate handling
    # ------------------------------------------------------------------

    @property
    def compression_model(self) -> BaseCompressionModel:
        """Return the compression model applied before every layer."""
        return self.rvq.vq.compression_model

    @property
    def num_codebooks(self) -> int:
        """Return the number of codebook layers."""
        return len(self.rvq.vq.layers)

    def set_inference_rate(self, rate) -> None:
        """Set the compression rate used when no explicit rate is passed.

        Needed at inference because ``codec.encode()`` does not forward
        extra kwargs down to the quantizer.  Pair every call with
        ``reset_inference_rate``.

        Args:
            rate: Compression rate (float or tensor, see ``forward``), or
                ``None`` to fall back to ``eval_rate``.
        """
        self._inference_rate = rate

    def reset_inference_rate(self) -> None:
        """Clear the rate set by ``set_inference_rate``."""
        self._inference_rate = None

    def set_inference_anchor_start_layer(self, layer: Optional[int]) -> None:
        """Set the anchor start layer used by encode when none is passed.

        Layers at/after this index constrain their boundaries to the union
        of the earlier (free) layers' boundaries.  Inference-only: like the
        inference rate, it exists because ``codec.encode()`` forwards no
        extra kwargs.  Reset after use so training is never affected.

        Args:
            layer: Index of the first anchored layer, or ``None`` to disable
                anchoring.
        """
        self._inference_anchor_start_layer = layer

    def reset_inference_anchor_start_layer(self) -> None:
        """Clear the layer set by ``set_inference_anchor_start_layer``."""
        self._inference_anchor_start_layer = None

    def _sample_rate(self, batch_size: int, device) -> Optional[torch.Tensor]:
        """Sample a training-time rate tensor according to ``random_rate``.

        Ported from the source repo's ``BaseCodecLightningModule._sample_rate``
        (generator-side variant; one sample per batch serves both GAN turns
        because SoundStream/Encodec cache generator outputs in training).
        """
        n_q = self.num_codebooks
        span = self.max_rate - self.min_rate

        if self.random_rate == "per_quantizer":
            return torch.rand(n_q, batch_size, device=device) * span + self.min_rate

        if self.random_rate == "per_sample":
            return torch.rand(batch_size, device=device) * span + self.min_rate

        # "none": no compression during training.
        return None

    def _resolve_rate(self, batch_size: int, device, rate):
        """Pick the rate: explicit, sampled (train), inference, then eval."""
        if rate is not None:
            return rate
        if self.training:
            return self._sample_rate(batch_size, device)
        if self._inference_rate is not None:
            return self._inference_rate
        return self.eval_rate

    # ------------------------------------------------------------------
    # ResidualVectorQuantizer interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ):
        """Residual vector quantization with per-layer compression.

        Mirrors ``ResidualVectorQuantizer.forward(x, sample_rate,
        bandwidth)``; the extra kwargs are only used when calling the
        wrapper directly (the codecs themselves never pass them).

        Args:
            x: (B, D, T) encoder output.
            sample_rate: Frame rate of ``x`` in Hz, as the codec passes it.
            bandwidth: Target bandwidth in kbps; ``None`` uses all layers.
            rate: Compression rate: a float, a (B,) tensor or a
                (num_codebooks, B) tensor.  ``None`` samples a rate in
                training mode and otherwise uses the inference rate, then
                ``eval_rate``.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: Layers at or after this index may only place
                boundaries inside the union of the earlier layers'
                boundaries.  ``None`` disables anchoring.

        Returns:
            Without quantizer dropout, the tuple ``(quantized, codes,
            bandwidth, commit_loss)``; with quantizer dropout (DAC), the
            tuple ``(quantized, codes, bandwidth, commit_loss,
            quantization_loss)``.  ``quantized`` is (B, D, T) and ``codes``
            is (n_q, B, T); the rest are scalar tensors.
        """
        rate = self._resolve_rate(x.size(0), x.device, rate)
        bandwidth_per_quantizer = self.rvq.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)

        if not self.rvq.quantizer_dropout:
            quantized, codes, commit_loss = self.rvq.vq(
                x,
                n_q=n_q,
                rate=rate,
                padding_mask=padding_mask,
                anchor_start_layer=anchor_start_layer,
            )
            used_bandwidth = torch.tensor(n_q * bandwidth_per_quantizer).to(x)
            return quantized, codes, used_bandwidth, torch.mean(commit_loss)
        else:
            quantized, codes, commit_loss, quantization_loss = self.rvq.vq(
                x,
                n_q=n_q,
                rate=rate,
                padding_mask=padding_mask,
                anchor_start_layer=anchor_start_layer,
            )
            used_bandwidth = torch.tensor(n_q * bandwidth_per_quantizer).to(x)
            return (
                quantized,
                codes,
                used_bandwidth,
                torch.mean(commit_loss),
                torch.mean(quantization_loss),
            )

    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode ``x`` to codes at the given bandwidth.

        Args:
            x: (B, D, T) encoder output.
            sample_rate: Frame rate of ``x`` in Hz, as the codec passes it.
            bandwidth: Target bandwidth in kbps; ``None`` uses all layers.
            st: Index of the first codebook layer to use; ``None`` means 0.
            rate: Compression rate; see ``forward``.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: See ``forward``; ``None`` uses the value set
                by ``set_inference_anchor_start_layer``.

        Returns:
            (n_q, B, T) code indices.
        """
        rate = self._resolve_rate(x.size(0), x.device, rate)
        if anchor_start_layer is None:
            anchor_start_layer = self._inference_anchor_start_layer
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        return self.rvq.vq.encode(
            x,
            n_q=n_q,
            st=st or 0,
            rate=rate,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.

        Args:
            codes: (n_q, B, T) code indices.

        Returns:
            (B, D, T) quantized representation.
        """
        return self.rvq.vq.decode(codes)

    def encode_with_segments(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
        rate: Optional[Union[float, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode and also return per-layer residuals and segment indices.

        Use this to measure the frame rate a compression rate actually
        produces: the segment indices give the number of segments per layer.

        Args:
            x: (B, D, T) encoder output.
            sample_rate: Frame rate of ``x`` in Hz, as the codec passes it.
            bandwidth: Target bandwidth in kbps; ``None`` uses all layers.
            st: Index of the first codebook layer to use; ``None`` means 0.
            rate: Compression rate; see ``forward``.
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.
            anchor_start_layer: See ``forward``; ``None`` uses the value set
                by ``set_inference_anchor_start_layer``.

        Returns:
            Tuple of the (n_q, B, T) code indices, the (n_q, B, D, T)
            residuals fed to each layer before compression, and a list with
            one (B, T) segment index tensor per layer.
        """
        rate = self._resolve_rate(x.size(0), x.device, rate)
        if anchor_start_layer is None:
            anchor_start_layer = self._inference_anchor_start_layer
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        return self.rvq.vq.encode_with_segments(
            x,
            n_q=n_q,
            st=st or 0,
            rate=rate,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )

    def get_num_quantizers_for_bandwidth(
        self, sample_rate: int, bandwidth: Optional[float] = None
    ) -> int:
        """Return the number of layers needed for ``bandwidth`` (kbps)."""
        return self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return the bandwidth of one codebook layer in kbps."""
        return self.rvq.get_bandwidth_per_quantizer(sample_rate)

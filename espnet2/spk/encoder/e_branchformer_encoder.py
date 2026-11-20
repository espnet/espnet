"""MFA E-Branchformer encoder for spoken language identification."""

from typing import Optional, Tuple

import torch

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder


class MfaEBranchformerEncoder(EBranchformerEncoder):
    """Multi-scale feature aggregation (MFA) over the ESPnet2 ASR E-Branchformer.

    As in MFA-Conformer (Zhang et al., INTERSPEECH 2022; ``MfaConformerEncoder``
    in ``espnet2/spk/encoder/conformer_encoder.py``), the outputs of all blocks
    are concatenated along the channels and layer-normalized, so that the
    pooling sees every depth (``output_size * num_blocks`` channels).

    Unlike the TDNN encoders of ``espnet2/spk``, it takes the frame lengths,
    masks the padded frames in self-attention, and returns the lengths after
    subsampling (``returns_lengths``), so that the pooling also skips the
    padded frames. The output is channel-first, as the spk/LID poolings expect.

    The keyword arguments are passed to ``EBranchformerEncoder``.
    ``layer_drop_rate`` must be 0, since every block contributes its channels.
    The final ``after_norm`` of ``EBranchformerEncoder`` is not used.

    Example:
        >>> encoder = MfaEBranchformerEncoder(80, output_size=256, num_blocks=6)
        >>> frames, lengths = encoder(feats, feat_lengths)
        >>> frames.shape[1] == encoder.output_size() == 256 * 6
        True
    """

    # ESPnetLIDModel passes the frame lengths to encoders with this flag and
    # uses the lengths they return for the pooling.
    returns_lengths = True

    def __init__(self, input_size: int, **kwargs):
        """Build the encoder and the layer norm of the concatenated blocks.

        Raises:
            ValueError: If ``layer_drop_rate`` is positive.
        """
        if kwargs.get("layer_drop_rate", 0.0) > 0:
            raise ValueError("MFA needs every block: layer_drop_rate must be 0")
        super().__init__(input_size, **kwargs)
        self._mfa_size = self._output_size * len(self.encoders)
        self.mfa_norm = torch.nn.LayerNorm(self._mfa_size)

    def output_size(self) -> int:
        """Return the number of channels of the concatenated blocks."""
        return self._mfa_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a padded batch of features.

        Args:
            x: Features (batch, frames, input_size).
            ilens: Valid frames of each utterance (batch,). None means that
                no utterance is padded.

        Returns:
            Encoded frames (batch, output_size * num_blocks, subsampled frames)
            and their valid lengths (batch,).
        """
        if ilens is None:
            ilens = torch.full(
                (x.size(0),), x.size(1), dtype=torch.long, device=x.device
            )
        (_, hidden), olens, _ = super().forward(x, ilens, return_all_hs=True)
        out = self.mfa_norm(torch.cat(hidden, dim=-1))
        return out.transpose(1, 2), olens

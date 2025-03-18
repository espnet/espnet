"""A simple linear layer decoder.

This can be used for classification tasks from sequence input.
"""

import logging
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LinearDecoder(AbsDecoder):

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        pooling: str = "mean",
        dropout: float = 0.0,
    ):
        """Initialize the module."""
        super().__init__()
        logging.warning(
            "Using Linear Decoder which is meant to be used for "
            "classification tasks only."
        )

        self.input_dim = encoder_output_size
        assert vocab_size > 3, "Invalid vocab size, must be > 3."
        self.output_dim = vocab_size - 3
        self.dropout = None
        if dropout != 0.0:
            self.dropout = torch.nn.Dropout(p=dropout)
        self.linear_out = torch.nn.Linear(self.input_dim, self.output_dim)
        assert pooling in [
            "mean",
            "max",
            "CLS",
        ], f"Invalid pooling: {pooling}. Should be 'mean', 'max' or 'CLS'."
        self.pooling = pooling

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor = None,
        ys_in_lens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method.

        Args:
            hs_pad: (B, Tmax, D)
            hlens: (B,)
        Returns:
            output: (B, n_classes)
        """

        mask = make_pad_mask(lengths=hlens, xs=hs_pad, length_dim=1).to(hs_pad.device)
        if self.dropout is not None:
            hs_pad = self.dropout(hs_pad)
        if self.pooling == "mean":
            unmasked_entries = (~mask).to(dtype=hs_pad.dtype)
            input_feature = (hs_pad * unmasked_entries).sum(dim=1)
            input_feature = input_feature / unmasked_entries.sum(dim=1)
        elif self.pooling == "max":
            input_feature = hs_pad.masked_fill(mask, float("-inf"))
            input_feature, _ = torch.max(input_feature, dim=1)
        elif self.pooling == "CLS":
            input_feature = hs_pad[:, 0, :]

        output = self.linear_out(input_feature)
        return output

    def score(self, ys, state, x):
        """Classify x.

        Args:
            ys: Not used
            state: Not used
            x: (T, D). this should be a single sample without
                any padding ie batch size=1.
        Returns:
            logp: log probabilities over (n_classes,)
            state: None
        Assumes that x is a single unpadded sequence.
        """
        hs_len = torch.tensor([x.shape[0]], dtype=torch.long).to(x.device)
        logits = self.forward(
            x.unsqueeze(0),
            hs_len,
        )
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        # Fix blank, unk and sos/eos to -inf
        minf_tensor = torch.tensor(float("-inf"), device=logp.device)
        minf_tensor = minf_tensor.expand(*(logp.shape[:-1]), 1)
        logp = torch.cat([minf_tensor, minf_tensor, logp, minf_tensor], dim=-1)
        return logp.squeeze(0), None

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim

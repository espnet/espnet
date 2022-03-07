# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
import logging
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder


class EnsembleDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.
    Args:
        decoders: ensemble decoders
    """

    def __init__(
        self,
        decoders: List[AbsDecoder],
        weights: List[float] = None,
    ):
        assert check_argument_types()
        super().__init__()
        assert len(decoders) > 0, "At least one decoder is needed for ensembling"

        # Note (jiatong): different from other'decoders
        self.decoders = torch.nn.ModuleList(decoders)
        self.weights = (
            [1 / len(decoders)] * len(decoders) if weights is None else weights
        )

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return [None] * len(self.decoders)

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dummy forward"""
        pass

    def score(self, ys, state, x):
        """Score."""
        assert len(x) == len(
            self.decoders
        ), "Num of encoder output does not match number of decoders"
        logps = []
        states = []
        for i in range(len(self.decoders)):
            ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
            sub_state = None if state is None else state[i]
            logp, sub_state = self.decoders[i].forward_one_step(
                ys.unsqueeze(0), ys_mask, x[i].unsqueeze(0), cache=sub_state
            )
            logps.append(self.weights[i] * logp.squeeze(0))
            states.append(sub_state)
        return torch.sum(torch.stack(logps, dim=0), dim=0), states

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (Union[torch.Tensor, List[torch.Tensor]]):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        n_batch = len(states)
        n_decoders = len(self.decoders)

        all_state_list = [[]] * n_batch
        logps = []
        for i in range(len(self.decoders)):
            decoder_batch = [states[h][i] for h in range(n_batch)]
            logp, state_list = self.decoders[i].batch_score(ys, decoder_batch, xs[i])
            for j in range(n_batch):
                all_state_list[j].append(state_list[j])
            logps.append(self.weights[i] * logp)

        return torch.sum(torch.stack(logps, dim=0), dim=0), all_state_list

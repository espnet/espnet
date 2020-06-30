"""ScorerInterface implementation for CTC."""

import numpy as np
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.scorer_interface import PartialScorerInterface


class CTCPrefixScorer_nemo(PartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc, eos: int):
        """Initialize class.

        Args:
            ctc (nemo torchscript Module): The CTC implementaiton.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
       # FIX ME input assert yuekai 
        logp = self.ctc(x.transpose(0,1).unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 28, 27, np)  # blank_id 28, eos_id 27
        return 0, self.impl.initial_state()

    def select_state(self, state, i):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search

        Returns:
            state: pruned state

        """
        sc, st = state
        return sc[i], st[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

"""ScorerInterface implementation for UASR."""

import numpy as np
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore, CTCPrefixScoreTH
from espnet.nets.scorers.ctc import CTCPrefixScorer


class UASRPrefixScorer(CTCPrefixScorer):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, eos: int):
        """Initialize class."""
        self.eos = eos

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        x[:, 0] = x[:, 0] - 100000000000  # simulate a no-blank CTC
        self.logp = (
            torch.nn.functional.log_softmax(x, dim=1).detach().squeeze(0).cpu().numpy()
        )
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        x[:, 0] = x[:, 0] - 100000000000  # simulate a no-blank CTC
        logp = torch.nn.functional.log_softmax(x, dim=1).unsqueeze(
            0
        )  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

"""Length bonus module."""
import torch

from espnet.nets.scorer_interface import ScorerInterface


class LengthBonus(ScorerInterface):
    """Length bonus in beam search."""

    def __init__(self, n_vocab: int):
        """Initialize class.

        Args:
            n_vocab (int): The number of tokens in vocabulary for beam search

        """
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and None

        """
        return torch.tensor([1.0], device=x.device, dtype=x.dtype).expand(self.n), None

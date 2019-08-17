import torch

from espnet.nets.scorer_interface import ScorerInterface


class LengthBonus(ScorerInterface):
    """Length bonus in beam search"""

    def __init__(self, n_vocab):
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token

        Args:
            y (torch.Tensor): torch.int64 prefix token to score (B)
            state: decoder state for prefix tokens
            x (torch.Tensor): Encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for y (B)
                and next state for ys
        """
        return torch.tensor([1.0], device=y.device).expand(self.n), None

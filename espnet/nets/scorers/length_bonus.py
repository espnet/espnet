import torch

from espnet.nets.scorer_interface import ScorerInterface


class LengthBonus(ScorerInterface):
    """Length bonus in beam search"""

    def __init__(self, n_vocab):
        self.n = n_vocab

    def score(self, y, state, x):
        return torch.tensor([1.0], device=x.device, dtype=x.dtype).expand(self.n), None
